#!/usr/bin/env python3
"""
NumPyro subprocess server for PPLBench.
Mirrors the Impulse harness.jl protocol: ###READY###/###DONE###/EXIT.
"""
import argparse
import importlib
import json
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_prng_impl", "rbg")

import jax.numpy as jnp
import numpy as np
import numpyro.infer as infer
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


def make_rng_key(seed):
    key_u64 = np.array([seed, 0], dtype=np.uint64)
    key_u32 = key_u64.view(np.uint32)
    return jax.random.wrap_key_data(key_u32, impl="rbg")


def make_init_params(raw_init):
    if raw_init is None:
        return None
    result = {}
    for k, v in raw_init.items():
        arr = jnp.array(v, dtype=jnp.float64)
        if arr.ndim == 1 and arr.shape[0] == 1:
            arr = arr.squeeze()
        result[k] = arr
    return result


def constrained_to_unconstrained(model_fn, data, constrained_init):
    """Convert constrained init_params to unconstrained using NumPyro's transforms."""
    if constrained_init is None:
        return None
    from numpyro.infer.util import initialize_model
    import numpyro.infer
    # Use a threefry key (not RBG) since initialize_model's seed handler
    # may not support RBG keys in all NumPyro versions.
    rng_key = jax.random.wrap_key_data(
        np.array([0, 0], dtype=np.uint32), impl="threefry2x32"
    )
    init_info, _, _, _ = initialize_model(
        rng_key, model_fn, model_args=(data,),
        init_strategy=numpyro.infer.init_to_value(values=constrained_init),
    )
    return init_info.z


def dump_mcmc_hlo(mcmc, rng_key, data, jax_init, module_name):
    """Dump the HLO of the full MCMC run (init + warmup + sampling).

    With progress_bar=False, NumPyro's mcmc.run compiles down to a single
    lax.scan program. We wrap mcmc.run in jax.jit and lower it to get the
    exact same HLO that XLA compiles during benchmarking.
    """
    dump_dir = os.path.join(
        os.environ.get("PPLBENCH_OUTPUT_DIR", "outputs"), "mlir_dumps", "numpyro"
    )
    os.makedirs(dump_dir, exist_ok=True)

    def mcmc_run(rng_key):
        mcmc.run(rng_key, data, init_params=jax_init)
        return mcmc.get_samples()

    print("NumPyro: Lowering mcmc.run for HLO dump...", flush=True)
    t0 = time.time()
    lowered = jax.jit(mcmc_run).lower(rng_key)
    hlo = lowered.as_text()
    dt = time.time() - t0

    dump_path = os.path.join(dump_dir, f"{module_name}_mcmc_kernel.hlo")
    with open(dump_path, "w") as f:
        f.write(hlo)
    sz = os.path.getsize(dump_path) // 1024
    print(f"NumPyro HLO: {dump_path} ({sz} KB, lowered in {dt:.2f}s)", flush=True)


def run_trial(mcmc, rng_key, data, jax_init):
    mcmc._last_state = None
    mcmc.run(rng_key, data, init_params=jax_init)
    jax.block_until_ready(mcmc.get_samples())
    return mcmc.get_samples()


def write_output(output_path, samples, impl, compile_time, run_time):
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
    mod = importlib.import_module(f"pplbench.ppls.numpyro.{module_path}")
    impl_class = getattr(mod, class_name)
    impl = impl_class(**data.attrs)

    position_size = None
    if init_params_raw:
        position_size = sum(np.array(v).size for v in init_params_raw.values())

    kernel = infer.NUTS(
        impl.model,
        step_size=args.step_size,
        adapt_step_size=args.adapt_step_size,
        adapt_mass_matrix=args.adapt_mass_matrix,
        max_tree_depth=args.max_tree_depth,
        dense_mass=True,
        find_heuristic_step_size=False,
        **({"inverse_mass_matrix": jnp.eye(position_size, dtype=jnp.float64)}
           if position_size else {}),
    )
    mcmc = infer.MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        progress_bar=False,
    )

    rng_key = make_rng_key(args.seed)
    jax_init = make_init_params(init_params_raw)
    jax_init = constrained_to_unconstrained(impl.model, data, jax_init)

    # JIT warmup
    print("NumPyro: JIT warmup...", flush=True)
    jit_start = time.time()
    run_trial(mcmc, rng_key, data, jax_init)
    jit_time = time.time() - jit_start
    print(f"NumPyro: JIT warmup done ({jit_time:.2f}s)", flush=True)

    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        dump_mcmc_hlo(mcmc, rng_key, data, jax_init, module_path.replace(".", "_"))

    # First real trial
    t0 = time.time()
    samples = run_trial(mcmc, rng_key, data, jax_init)
    run_time = time.time() - t0

    write_output(args.output, samples, impl, jit_time, run_time)
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
            new_init = constrained_to_unconstrained(impl.model, data, new_init)

            new_rng = make_rng_key(new_seed)
            profile_dir = req.get("profile")

            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
                print(f"Profiling -> {profile_dir}", flush=True)
                with jax.profiler.trace(profile_dir, create_perfetto_link=False):
                    t0 = time.time()
                    new_samples = run_trial(mcmc, new_rng, data, new_init)
                    new_run_time = time.time() - t0
            else:
                t0 = time.time()
                new_samples = run_trial(mcmc, new_rng, data, new_init)
                new_run_time = time.time() - t0

            write_output(new_output_path, new_samples, impl, 0.0, new_run_time)
            print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
