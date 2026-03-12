#!/usr/bin/env python3
"""
PyMC subprocess server for PPLBench.
Mirrors the NumPyro harness protocol: ###READY###/###DONE###/EXIT.
"""
import argparse
import importlib
import json
import os
import sys
import time

# Set LD_LIBRARY_PATH for openblas before any PyTensor import
conda_lib = os.environ.get("PPLBENCH_CONDA_LIB", "")
if conda_lib:
    os.environ["LD_LIBRARY_PATH"] = conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    import ctypes
    ctypes.CDLL(os.path.join(conda_lib, "libopenblas.so"), mode=ctypes.RTLD_GLOBAL)

import numpy as np
import xarray as xr


def reconstruct_dataset(data_dict):
    data_vars = {}
    dims = data_dict.get("dims", {})
    coords_raw = data_dict.get("coords", {})
    attrs = data_dict.get("attrs", {})

    for var in dims:
        data_vars[var] = (dims[var], np.array(data_dict[var]))

    coords = {c: np.array(v) for c, v in coords_raw.items()}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def make_init_params(raw_init):
    if raw_init is None:
        return None
    result = {}
    for k, v in raw_init.items():
        arr = np.array(v, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == 1:
            arr = arr.squeeze()
        result[k] = arr
    return result


def run_trial(model, num_warmup, num_samples, seed, step_size,
              max_tree_depth, adapt_step_size, adapt_mass_matrix,
              init_params=None):
    import pymc as pm

    with model:
        step_kwargs = {
            "max_treedepth": max_tree_depth,
            "target_accept": 0.8,
        }
        if not adapt_step_size:
            step_kwargs["step_scale"] = step_size

        step = pm.NUTS(**step_kwargs)

        initvals = None
        if init_params is not None:
            initvals = {}
            for name, val in init_params.items():
                initvals[name] = val

        trace = pm.sample(
            draws=num_samples,
            tune=num_warmup,
            step=step,
            chains=1,
            cores=1,
            random_seed=seed,
            initvals=initvals,
            progressbar=False,
            discard_tuned_samples=True,
            return_inferencedata=True,
        )

    samples = {}
    posterior = trace.posterior
    for var_name in posterior.data_vars:
        vals = posterior[var_name].values
        # Shape is (chains, draws, ...) -> squeeze chain dim
        vals = vals.squeeze(axis=0)
        samples[var_name] = vals

    return samples


def write_output(output_path, samples, compile_time, run_time):
    output = {}
    for k, v in samples.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            output[k] = arr.tolist()
        else:
            output[k] = [arr[i].tolist() for i in range(arr.shape[0])]
    output["compile_time"] = compile_time
    output["run_time"] = run_time

    with open(output_path, "w") as f:
        json.dump(output, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--adapt-step-size", action="store_true")
    parser.add_argument("--adapt-mass-matrix", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    with open(args.data) as f:
        data_dict = json.load(f)

    data = reconstruct_dataset(data_dict)
    init_params_raw = data_dict.get("init_params")

    module_path, class_name = args.model_class.rsplit(".", 1)
    mod = importlib.import_module(f"pplbench.ppls.pymc.{module_path}")
    impl_class = getattr(mod, class_name)
    impl = impl_class(**data.attrs)

    # Build the PyMC model
    print("PyMC: Building model...", flush=True)
    compile_start = time.time()
    pymc_model = impl.build_model(data)
    compile_time = time.time() - compile_start
    print(f"PyMC: Model built ({compile_time:.2f}s)", flush=True)

    # First real trial
    jax_init = make_init_params(init_params_raw)
    t0 = time.time()
    samples = run_trial(
        pymc_model, args.num_warmup, args.num_samples, args.seed,
        args.step_size, args.max_tree_depth,
        args.adapt_step_size, args.adapt_mass_matrix,
        init_params=jax_init,
    )
    run_time = time.time() - t0

    write_output(args.output, samples, compile_time, run_time)
    print(f"First trial: {run_time:.3f}s", flush=True)

    if args.server:
        print("###READY###", flush=True)

        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "EXIT":
                break

            req = json.loads(stripped)
            new_seed = req["seed"]
            new_output_path = req["output"]
            new_init = make_init_params(req.get("init_params"))

            t0 = time.time()
            new_samples = run_trial(
                pymc_model, args.num_warmup, args.num_samples, new_seed,
                args.step_size, args.max_tree_depth,
                args.adapt_step_size, args.adapt_mass_matrix,
                init_params=new_init,
            )
            new_run_time = time.time() - t0

            write_output(new_output_path, new_samples, 0.0, new_run_time)
            print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
