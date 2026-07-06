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


def _early_num_chains():
    for i, a in enumerate(sys.argv):
        if a == "--num-chains" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 1


# `--xla_force_host_platform_device_count` must be set before jax initializes so
# chain_method="parallel" gets a genuine pmap over host devices (the H1 test
# pattern) rather than falling back to sequential. Only forced for num_chains>1.
_early_nc = _early_num_chains()
if _early_nc > 1 and "xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={_early_nc}"
    ).strip()

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_prng_impl", "rbg")

import jax.numpy as jnp
import numpy as np
import numpyro
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


def run_trial(mcmc, rng_key, data, jax_init, group_by_chain=False):
    mcmc._last_state = None
    mcmc.run(rng_key, data, init_params=jax_init)
    jax.block_until_ready(mcmc.get_samples())
    return mcmc.get_samples(group_by_chain=group_by_chain)


# =============================================================================
# SVI support
# =============================================================================

def _svi_init_constrained(model, data, seed):
    """constrain(0) at each latent site: exp(0)=1 for positive, 0 for real.

    Mirrors the Impulse harness's deterministic unconstrained guide loc = 0 so
    both backends start SVI from the same point (AutoNormal unconstrains these
    values internally to set the guide location)."""
    import numpyro.handlers as handlers
    from numpyro.distributions.transforms import biject_to

    # Only the site supports (shapes) are read here. Use a typed threefry key:
    # numpyro's seed handler rejects raw RBG keys produced from an int seed under
    # the global RBG prng impl (same reason `constrained_to_unconstrained` above
    # uses threefry).
    key = jax.random.wrap_key_data(
        np.array([0, 0], dtype=np.uint32), impl="threefry2x32"
    )
    tr = handlers.trace(handlers.seed(model, key)).get_trace(data)
    init = {}
    for name, site in tr.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            transform = biject_to(site["fn"].support)
            init[name] = transform(jnp.zeros(site["fn"].shape()))
    return init


def _build_svi(model, data, guide_name, optimizer_name, step_size,
               num_particles, init_scale, seed):
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta, AutoNormal
    from numpyro.infer.initialization import init_to_value

    init_c = _svi_init_constrained(model, data, seed)
    init_loc_fn = init_to_value(values=init_c)
    if guide_name == "AutoDelta":
        guide = AutoDelta(model, init_loc_fn=init_loc_fn)
    elif guide_name in ("AutoNormal", "AutoMultivariateNormal"):
        # AutoMultivariateNormal is validated separately; the SVI benchmark
        # models use AutoNormal, matching the Impulse configs.
        guide = AutoNormal(model, init_loc_fn=init_loc_fn, init_scale=init_scale)
    else:
        raise ValueError(f"Unsupported guide: {guide_name}")

    opts = {
        "Adam": lambda: numpyro.optim.Adam(step_size=step_size),
        "SGD": lambda: numpyro.optim.SGD(step_size=step_size),
        "ClippedAdam": lambda: numpyro.optim.ClippedAdam(step_size=step_size),
        "Momentum": lambda: numpyro.optim.Momentum(step_size=step_size, mass=0.9),
        "RMSProp": lambda: numpyro.optim.RMSProp(step_size=step_size),
        "Adagrad": lambda: numpyro.optim.Adagrad(step_size=step_size),
        "RMSPropMomentum":
            lambda: numpyro.optim.RMSPropMomentum(step_size=step_size),
    }
    if optimizer_name not in opts:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    opt = opts[optimizer_name]()

    svi = SVI(model, guide, opt, loss=Trace_ELBO(num_particles=num_particles))
    return svi, guide


def _svi_write_diagnostics(model_name, seed, elbo_traj, per_step, num_steps):
    diag_dir = os.path.join(
        os.environ.get("PPLBENCH_OUTPUT_DIR", "outputs"), "svi_diagnostics"
    )
    os.makedirs(diag_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(model_name))
    path = os.path.join(diag_dir, f"numpyro_{safe}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump({
            "model": str(model_name),
            "seed": int(seed),
            "num_steps": int(num_steps),
            "per_step_s": float(per_step),
            "final_elbo": float(elbo_traj[-1]) if len(elbo_traj) else float("nan"),
            "elbo_trajectory": [float(x) for x in elbo_traj],
        }, f)


def run_svi_trial(svi, guide, rng_key, data, num_steps, num_draws):
    """Fit for num_steps (timed) then draw num_draws from the fitted guide via
    sample_posterior (Predictive over the guide, constrained space). Only the fit
    is timed, matching the Impulse harness (guide draws are a post-fit diagnostic)."""
    fit_key, draw_key = jax.random.split(rng_key)
    t0 = time.time()
    res = svi.run(fit_key, num_steps, data, progress_bar=False)
    res.losses.block_until_ready()
    run_time = time.time() - t0
    params = res.params
    losses = np.asarray(res.losses)
    draws = guide.sample_posterior(draw_key, params, sample_shape=(num_draws,))
    jax.block_until_ready(draws)
    samples = {k: np.asarray(v) for k, v in draws.items()}
    return samples, (-losses).tolist(), run_time


def svi_main(args, impl, data, module_path):
    # NumPyro's AutoGuide internally seeds with an int (rng_seed=0), which the
    # seed handler rejects under the global RBG prng impl this harness uses for
    # MCMC. SVI parity with Impulse is not bitwise (guide draws / gradient
    # sampling differ), so use threefry here, matching the D3 SVI reference.
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    model_name = module_path
    K = args.num_samples

    def one_trial(seed):
        rng_key = jax.random.PRNGKey(int(seed))
        svi, guide = _build_svi(
            impl.model, data, args.guide, args.optimizer, args.step_size,
            args.num_particles, args.init_scale, seed,
        )
        # JIT warm (discarded), then a timed fit.
        run_svi_trial(svi, guide, rng_key, data, args.num_steps, K)
        samples, elbo_traj, run_time = run_svi_trial(
            svi, guide, rng_key, data, args.num_steps, K
        )
        _svi_write_diagnostics(
            model_name, seed, elbo_traj, run_time / args.num_steps, args.num_steps
        )
        return samples, elbo_traj, run_time

    print("NumPyro SVI: JIT warmup + first trial...", flush=True)
    jit_start = time.time()
    samples, elbo_traj, run_time = one_trial(args.seed)
    jit_time = time.time() - jit_start
    write_output(args.output, samples, impl, jit_time, run_time)
    final = elbo_traj[-1] if elbo_traj else float("nan")
    print(f"First trial: {run_time:.3f}s final_elbo={final:.6f}", flush=True)

    if not args.server:
        return

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
        new_samples, _, new_run_time = one_trial(new_seed)
        write_output(new_output_path, new_samples, impl, 0.0, new_run_time)
        print(f"###DONE### {new_run_time * 1000:.2f} ms", flush=True)


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
    parser.add_argument("--method", default="nuts")
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--chain-method", default="parallel")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-particles", type=int, default=1)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--guide", default="AutoNormal")
    parser.add_argument("--optimizer", default="Adam")
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

    if args.method == "svi":
        svi_main(args, impl, data, module_path)
        return

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
    num_chains = args.num_chains
    group_by_chain = num_chains > 1
    mcmc = infer.MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=num_chains,
        chain_method=args.chain_method,
        progress_bar=False,
    )
    if group_by_chain:
        ndev = jax.local_device_count()
        print(
            f"NumPyro: num_chains={num_chains}, chain_method={args.chain_method}, "
            f"local_device_count={ndev} "
            f"({'genuine pmap' if ndev >= num_chains else 'FALLBACK to sequential'})",
            flush=True,
        )

    rng_key = make_rng_key(args.seed)

    def prep_init(raw):
        # Multi-chain init_params carry a leading num_chains dim (numpyro splits
        # the key per chain; same init position for all chains matches the
        # Impulse arm, which feeds every chain the same trial constraint).
        j = make_init_params(raw)
        j = constrained_to_unconstrained(impl.model, data, j)
        if group_by_chain and j is not None:
            j = {k: jnp.broadcast_to(v, (num_chains,) + jnp.shape(v))
                 for k, v in j.items()}
        return j

    jax_init = prep_init(init_params_raw)

    # JIT warmup
    print("NumPyro: JIT warmup...", flush=True)
    jit_start = time.time()
    run_trial(mcmc, rng_key, data, jax_init, group_by_chain)
    jit_time = time.time() - jit_start
    print(f"NumPyro: JIT warmup done ({jit_time:.2f}s)", flush=True)

    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        dump_mcmc_hlo(mcmc, rng_key, data, jax_init, module_path.replace(".", "_"))

    # First real trial
    t0 = time.time()
    samples = run_trial(mcmc, rng_key, data, jax_init, group_by_chain)
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
            new_init = prep_init(req.get("init_params"))

            new_rng = make_rng_key(new_seed)
            profile_dir = req.get("profile")

            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
                print(f"Profiling -> {profile_dir}", flush=True)
                with jax.profiler.trace(profile_dir, create_perfetto_link=False):
                    t0 = time.time()
                    new_samples = run_trial(mcmc, new_rng, data, new_init, group_by_chain)
                    new_run_time = time.time() - t0
            else:
                t0 = time.time()
                new_samples = run_trial(mcmc, new_rng, data, new_init, group_by_chain)
                new_run_time = time.time() - t0

            write_output(new_output_path, new_samples, impl, 0.0, new_run_time)
            print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
