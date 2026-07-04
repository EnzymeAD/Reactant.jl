#!/usr/bin/env python3
"""
Stan/CmdStanPy subprocess server for PPLBench.
Mirrors the Impulse harness.jl protocol: ###READY###/###DONE###/EXIT.
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import time

import numpy as np
import xarray as xr
from cmdstanpy import CmdStanModel


def reconstruct_dataset(data_dict):
    data_vars = {}
    dims = data_dict.get("dims", {})
    coords_raw = data_dict.get("coords", {})
    attrs = data_dict.get("attrs", {})

    for var in dims:
        data_vars[var] = (dims[var], np.array(data_dict[var]))

    coords = {c: np.array(v) for c, v in coords_raw.items()}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def run_trial(model, stan_data, num_warmup, num_samples, seed,
              step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
              init_params):
    inits = None
    if init_params is not None:
        inits = {}
        for k, v in init_params.items():
            arr = np.array(v)
            inits[k] = arr.item() if arr.ndim == 1 and arr.shape[0] == 1 else arr

    fit = model.sample(
        data=stan_data,
        chains=1,
        iter_warmup=num_warmup,
        iter_sampling=num_samples,
        seed=seed,
        step_size=step_size,
        metric='dense_e',
        max_treedepth=max_tree_depth,
        adapt_engaged=adapt_step_size or adapt_mass_matrix,
        inits=inits,
        show_console=False,
        show_progress=False,
    )
    return fit.stan_variables()


def write_output(output_path, samples, impl, compile_time, run_time):
    result = impl.extract_data_from_stan(samples)

    output = {}
    for var in result.data_vars:
        vals = result[var].values
        if vals.ndim == 1:
            output[var] = vals.tolist()
        else:
            output[var] = [vals[i].tolist() for i in range(vals.shape[0])]
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

    module_path, class_name = args.model_class.rsplit(".", 1)
    mod = importlib.import_module(f"pplbench.ppls.stan.{module_path}")
    impl_class = getattr(mod, class_name)
    impl = impl_class(**data.attrs)

    stan_data = impl.format_data_to_stan(data)

    # Write Stan code to temp file and compile (AOT)
    stan_code = impl.get_code()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".stan", delete=False, prefix="pplbench_stan_"
    ) as f_stan:
        f_stan.write(stan_code)
        stan_file = f_stan.name

    try:
        print("Stan: compiling model...", flush=True)
        compile_start = time.time()
        stan_model = CmdStanModel(stan_file=stan_file)
        compile_time = time.time() - compile_start
        print(f"Stan: compilation done ({compile_time:.2f}s)", flush=True)

        # First trial
        t0 = time.time()
        samples = run_trial(
            stan_model, stan_data,
            args.num_warmup, args.num_samples, args.seed,
            args.step_size, args.max_tree_depth,
            args.adapt_step_size, args.adapt_mass_matrix,
            data_dict.get("init_params"),
        )
        run_time = time.time() - t0

        write_output(args.output, samples, impl, compile_time, run_time)
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
                new_init = req.get("init_params")

                t0 = time.time()
                new_samples = run_trial(
                    stan_model, stan_data,
                    args.num_warmup, args.num_samples, new_seed,
                    args.step_size, args.max_tree_depth,
                    args.adapt_step_size, args.adapt_mass_matrix,
                    new_init,
                )
                new_run_time = time.time() - t0

                write_output(new_output_path, new_samples, impl, 0.0, new_run_time)
                print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)

    finally:
        if os.path.exists(stan_file):
            os.unlink(stan_file)


if __name__ == "__main__":
    main()
