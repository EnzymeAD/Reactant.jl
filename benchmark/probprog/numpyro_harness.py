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


def check_matched_init_differentiable(model_fn, data, z):
    """Fail loudly if the reference gradient at the matched init is a tie-break.

    The `--validate` gate is only meaningful if the NumPyro arm's gradient at the
    shared starting point is the actual derivative. It is not, if that point sits
    on a branch point of the reference log-prob's algebraic form.

    This has happened once and cost a full investigation: numpyro's
    `Bernoulli(logits=x).log_prob(y)` routes through
      binary_cross_entropy_with_logits(x, y) = clip(x, 0) + log1p(exp(-|x|)) - x*y
    whose two pieces are each non-differentiable at x = 0, and JAX resolves them
    with inconsistent conventions (clip'(0) = 0.5 centred, |x|'(0) = +1
    one-sided). The kinks fail to cancel, so the reported gradient of log p is
    `y` rather than the true `y - 0.5`. Any logit that is a linear map of
    zero-initialised real-support sites is exactly 0, so a canonical all-zero
    init puts every observation on it, and the gate reports the resulting
    trajectory difference as an Impulse defect.

    THE CHECK. Displace every coordinate by d and recompute the gradient, for d
    over a range of vanishingly small magnitudes. Any d here is far below the ULP
    of every coordinate and of every intermediate, so a genuinely differentiable
    log-density returns a bit-identical gradient (the induced change is bounded by
    |Hessian| * d, roughly 1e-94 for these models). A branch point instead flips
    the tie-break and changes the gradient by O(1). That gives a separation of
    ~90 orders of magnitude, so the threshold below needs no tuning; it is set
    only loose enough to tolerate a benign tie in a numerical-stability `max`
    (logsumexp's amax, which is stop_gradient'ed and measures exactly 0 here).

    Several magnitudes are used because a single one is not robust: a displacement
    must survive the model's own arithmetic to be observable. 1 ULP (5e-324 at the
    origin) is annihilated by `X @ beta`, and 1e-300 would be annihilated by a
    model that rescales its inputs sharply. The largest surviving magnitude wins.

    SCOPE. This detects a kink AT a coordinate's current value, which is the
    structural case that actually bites: a canonical zero-valued site feeding a
    log-prob whose kink sits at the origin. It is silent for a coordinate that is
    already nonzero, where a kink would require a numerical coincidence
    (noisy_or_topic's Bernoulli logits sit 0.57 away from theirs).

    The fix when this fires is to offset the offending real-support site off the
    origin on BOTH arms (base_numpyro_impl.MATCHED_INIT_REAL and harness.jl
    MATCHED_INIT_REAL), never to relax the gate's tolerances.
    """
    from numpyro.infer.util import potential_energy

    def U(zz):
        return potential_energy(model_fn, (data,), {}, zz)

    g0 = jax.grad(U)(z)
    worst, worst_site, worst_d = 0.0, None, None
    for mag in (1e-100, 1e-200, 1e-300):
        for d in (mag, -mag):
            gp = jax.grad(U)({k: v + d for k, v in z.items()})
            for k in z:
                delta = float(jnp.max(jnp.abs(g0[k] - gp[k])))
                if delta > worst:
                    worst, worst_site, worst_d = delta, k, d
    scale = max(1.0, max(float(jnp.max(jnp.abs(g0[k]))) for k in z))
    print(f"NumPyro: matched-init branch-point check: max gradient change under "
          f"an infinitesimal displacement = {worst:.6e} (gradient scale "
          f"{scale:.6e}, worst site '{worst_site}', d={worst_d})", flush=True)
    if worst > 1e-8 * scale:
        raise RuntimeError(
            f"Matched init sits on a non-differentiable point of the NumPyro "
            f"reference log-prob: displacing the init by {worst_d} changes dU/dz "
            f"by {worst:.6e} at site '{worst_site}' (gradient scale "
            f"{scale:.6e}). The gradient at this point is a tie-break, not a "
            f"derivative, so --validate would compare Impulse against a wrong "
            f"reference. Offset the affected real-support site off the origin on "
            f"BOTH arms using MATCHED_INIT_REAL; do NOT loosen the validate "
            f"tolerances."
        )


def make_compiled_run(mcmc, data, group_by_chain=False):
    """The whole MCMC run (kernel init + warmup + sampling) as one function of
    the trial's RNG key and initial point, wrapped in jax.jit ONCE and reused by
    every trial.

    Why: numpyro 0.19 `util.fori_collect` (progress_bar=False) wraps a fresh
    `loop_fn` closure, with the initial sampler state baked in, in jax.jit on
    every `MCMC.run`, so calling `mcmc.run` per trial re-traces the model and
    recompiles the whole sampler inside every timed trial (3.5 to 5.3 s per
    trial on the continuous models, about 48 s on noisy-or; the 2026-10-03
    figure-11 check, oopsla/evidence/speedup_prediction_2026-10-03/
    fig11_crossframework, finding F1). The other three systems compile outside
    their timed span. Wrapping `mcmc.run` in jax.jit (the construction
    `dump_mcmc_hlo` lowers) gives the same kernel, settings, key and init in one
    program that is compiled by the untimed warm-up call and then reused.

    The initial point is a traced argument (not closed over), so a config that
    sends a different init per trial (no --matched-init, e.g. the GP row) also
    compiles once; under --matched-init every trial passes the same point."""

    def mcmc_run(rng_key, init_params):
        mcmc.run(rng_key, data, init_params=init_params)
        return mcmc.get_samples(group_by_chain=group_by_chain)

    return jax.jit(mcmc_run)


def dump_mcmc_hlo(compiled_run, rng_key, jax_init, module_name):
    """Dump the HLO of the full MCMC run (init + warmup + sampling): the
    lowering of the exact jitted program the timed trials execute."""
    dump_dir = os.path.join(
        os.environ.get("PPLBENCH_OUTPUT_DIR", "outputs"), "mlir_dumps", "numpyro"
    )
    os.makedirs(dump_dir, exist_ok=True)

    print("NumPyro: Lowering mcmc.run for HLO dump...", flush=True)
    t0 = time.time()
    lowered = compiled_run.lower(rng_key, jax_init)
    hlo = lowered.as_text()
    dt = time.time() - t0

    dump_path = os.path.join(dump_dir, f"{module_name}_mcmc_kernel.hlo")
    with open(dump_path, "w") as f:
        f.write(hlo)
    sz = os.path.getsize(dump_path) // 1024
    print(f"NumPyro HLO: {dump_path} ({sz} KB, lowered in {dt:.2f}s)", flush=True)


def run_trial(mcmc, rng_key, data, jax_init, group_by_chain=False):
    """Eager per-call `mcmc.run` (recompiles the sampler on every call, see
    `make_compiled_run`). Not used by `main()`; kept because the 2026-10-03
    check's probes (fig11_crossframework/scripts/numpyro_probe.py) time it as
    the as-run reference."""
    mcmc._last_state = None
    mcmc.run(rng_key, data, init_params=jax_init)
    jax.block_until_ready(mcmc.get_samples())
    return mcmc.get_samples(group_by_chain=group_by_chain)


def run_compiled_trial(compiled_run, rng_key, jax_init):
    """One timed trial: execute the once-compiled MCMC program and wait for
    the draws (the same span `run_trial` timed, without the recompilation)."""
    samples = compiled_run(rng_key, jax_init)
    jax.block_until_ready(samples)
    return samples


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


def _svi_guide_loc_scale(params):
    """Per-site AutoNormal guide params in UNCONSTRAINED latent space.

    NumPyro's AutoNormal names its params ``{site}_auto_loc`` (guide mean) and
    ``{site}_auto_scale`` (softplus-positive std). Return two dicts keyed by the
    bare site name so the two harnesses compare by NAME rather than column
    position; this sidesteps the declaration-vs-alphabetical ordering issue that
    bit the n_schools guide draws (declaration-first sigma_* sort last)."""
    loc, scale = {}, {}
    for k, v in params.items():
        arr = [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]
        if k.endswith("_auto_loc"):
            loc[k[: -len("_auto_loc")]] = arr
        elif k.endswith("_auto_scale"):
            scale[k[: -len("_auto_scale")]] = arr
    return loc, scale


def _svi_write_diagnostics(model_name, seed, elbo_traj, per_step, num_steps,
                           guide_loc=None, guide_scale=None, svi_init_s=None):
    diag_dir = os.path.join(
        os.environ.get("PPLBENCH_OUTPUT_DIR", "outputs"), "svi_diagnostics"
    )
    os.makedirs(diag_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(model_name))
    path = os.path.join(diag_dir, f"numpyro_{safe}_seed{seed}.json")
    payload = {
        "model": str(model_name),
        "seed": int(seed),
        "num_steps": int(num_steps),
        "per_step_s": float(per_step),
        "final_elbo": float(elbo_traj[-1]) if len(elbo_traj) else float("nan"),
        "elbo_trajectory": [float(x) for x in elbo_traj],
    }
    # Share of the timed fit spent in svi.init (eager guide/model trace that
    # builds the per-seed SVIState); the rest is the compiled update scan.
    if svi_init_s is not None:
        payload["svi_init_s"] = float(svi_init_s)
    # Fitted guide loc/scale (validate correctness compares the FIT, not draws).
    if guide_loc is not None:
        payload["loc"] = guide_loc
        payload["scale"] = guide_scale
    with open(path, "w") as f:
        json.dump(payload, f)


def make_compiled_svi_fit(svi, data, num_steps):
    """`num_steps` SVI updates as one lax.scan, wrapped in jax.jit ONCE.

    Why: `svi.run(progress_bar=False)` runs an eager lax.scan over a fresh body
    closure, so every call re-traces the model and guide and recompiles the
    step program inside the timed fit (2.2 to 5.7 s per fit against fits of 0.18
    to 2.7 s; the 2026-10-03 figure-11 check, finding F5). This is the scan
    `svi.run` builds (`self.update(svi_state, *args)` per step, reverse-mode,
    no stable_update), compiled by the untimed warm-up fit and reused."""

    def fit(svi_state):
        def body(state, _):
            return svi.update(state, data)

        return jax.lax.scan(body, svi_state, None, length=num_steps)

    return jax.jit(fit)


def run_compiled_svi_trial(svi, guide, compiled_fit, rng_key, data, num_draws,
                           matched=False):
    """One timed fit with the step program compiled once: `svi.init` (the
    per-seed SVIState, as `svi.run` creates it) plus the compiled scan of
    updates, i.e. the work `run_svi_trial` timed without the re-trace and
    recompilation. Then draw num_draws from the fitted guide (untimed). Key
    handling is `run_svi_trial`'s (see there)."""
    if matched:
        fit_key = rng_key
        draw_key = jax.random.split(rng_key)[0]
    else:
        fit_key, draw_key = jax.random.split(rng_key)
    t0 = time.time()
    svi_state = svi.init(fit_key, data)
    jax.block_until_ready(svi_state)
    init_time = time.time() - t0
    svi_state, losses = compiled_fit(svi_state)
    losses.block_until_ready()
    run_time = time.time() - t0
    params = svi.get_params(svi_state)
    losses = np.asarray(losses)
    draws = guide.sample_posterior(draw_key, params, sample_shape=(num_draws,))
    jax.block_until_ready(draws)
    samples = {k: np.asarray(v) for k, v in draws.items()}
    return samples, (-losses).tolist(), run_time, params, init_time


def run_svi_trial(svi, guide, rng_key, data, num_steps, num_draws, matched=False):
    """As-run reference (`svi.run` per call, which recompiles every call; see
    `make_compiled_svi_fit`). Not used by `svi_main`; kept because the
    2026-10-03 check's probes time it.

    Fit for num_steps (timed) then draw num_draws from the fitted guide via
    sample_posterior (Predictive over the guide, constrained space). Only the fit
    is timed, matching the Impulse harness (guide draws are a post-fit diagnostic).

    When `matched` (validate correctness mode), the fit consumes `rng_key`
    DIRECTLY (an rbg key from make_rng_key) so the reparameterization-noise stream
    is byte-identical to the Impulse SVI arm, whose enzyme.random DEFAULT algorithm
    is jax's rbg generator seeded from [seed, 0]. The draw key is derived
    separately (guide draws are not part of the compared FIT). The normal
    benchmark path keeps the historical split(rng_key) -> (fit_key, draw_key)."""
    if matched:
        fit_key = rng_key
        draw_key = jax.random.split(rng_key)[0]
    else:
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
    return samples, (-losses).tolist(), run_time, params


def svi_main(args, impl, data, module_path):
    # NumPyro's AutoGuide internally seeds with an int (rng_seed=0), which the
    # seed handler rejects under the global RBG prng impl this harness uses for
    # MCMC. SVI parity with Impulse is not bitwise (guide draws / gradient
    # sampling differ), so use threefry here, matching the D3 SVI reference.
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    model_name = module_path
    K = args.num_samples

    # SVI, guide and the compiled fit are built ONCE per server (the guide's
    # init point and every setting are seed-independent; only the key varies
    # per trial). The first fit below compiles the step program (untimed).
    svi, guide = _build_svi(
        impl.model, data, args.guide, args.optimizer, args.step_size,
        args.num_particles, args.init_scale, args.seed,
    )
    compiled_fit = make_compiled_svi_fit(svi, data, args.num_steps)

    def one_trial(seed, warm=False):
        # Validate correctness (`--matched-init`): seed the fit with the rbg key
        # from [seed, 0] (make_rng_key) so the reparam-noise stream matches the
        # Impulse SVI arm exactly. Normal benchmark path keeps threefry PRNGKey.
        rng_key = (make_rng_key(int(seed)) if args.matched_init
                   else jax.random.PRNGKey(int(seed)))
        if warm:
            # JIT warm-up fit (discarded): compiles the step program once.
            run_compiled_svi_trial(svi, guide, compiled_fit, rng_key, data, K,
                                   matched=args.matched_init)
        samples, elbo_traj, run_time, params, init_time = run_compiled_svi_trial(
            svi, guide, compiled_fit, rng_key, data, K, matched=args.matched_init
        )
        guide_loc, guide_scale = _svi_guide_loc_scale(params)
        _svi_write_diagnostics(
            model_name, seed, elbo_traj, run_time / args.num_steps, args.num_steps,
            guide_loc=guide_loc, guide_scale=guide_scale, svi_init_s=init_time,
        )
        return samples, elbo_traj, run_time

    print("NumPyro SVI: JIT warmup + first trial...", flush=True)
    jit_start = time.time()
    samples, elbo_traj, run_time = one_trial(args.seed, warm=True)
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
    parser.add_argument("--matched-init", action="store_true")
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

    # Matched-init control experiment: replace the default prior/z=0 init with a
    # deterministic model-specific constrained point (uniform simplices) provided
    # by the impl. This is a CONSTRAINED init routed through init_to_value below,
    # so its per-variable sizes are the constrained event sizes and must NOT feed
    # the unconstrained inverse_mass_matrix override (`position_size`). The kernel
    # is otherwise unchanged vs the standard protocol; only the start point moves.
    matched_fn = getattr(impl, "matched_init_params", None)
    matched = bool(args.matched_init and matched_fn is not None)
    if matched:
        # Matched init overrides any driver-supplied per-trial init_params (a
        # model with generate_init_params, e.g. the regression models, sends a
        # per-trial Uniform(-2,2) draw; matched-init must ignore it so every trial
        # and every system starts from the same deterministic point).
        init_params_raw = {k: np.asarray(v).tolist()
                           for k, v in matched_fn(data).items()}
        print("NumPyro: --matched-init constrained point: "
              + ", ".join(f"{k}={np.asarray(v).reshape(-1)[:3]}"
                          for k, v in init_params_raw.items()), flush=True)

    position_size = None
    if init_params_raw and not matched:
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

    if matched and jax_init is not None:
        max_absz = max(float(jnp.max(jnp.abs(v))) for v in jax_init.values())
        tot = sum(int(jnp.size(v)) for v in jax_init.values())
        print(f"NumPyro: matched-init unconstrained z: dim={tot}, "
              f"max|z|={max_absz:.3e} (0 == symmetric point)", flush=True)
        # The gate is vacuous if the reference gradient here is a tie-break.
        # `jax_init` may carry a leading chain dim; check the single-chain point.
        check_matched_init_differentiable(
            impl.model, data,
            {k: (v[0] if group_by_chain else v) for k, v in jax_init.items()},
        )

    # The whole sampler as one program, compiled once by the untimed JIT
    # warmup below and reused by every timed trial (see make_compiled_run).
    compiled_run = make_compiled_run(mcmc, data, group_by_chain)

    # JIT warmup (untimed)
    print("NumPyro: JIT warmup...", flush=True)
    jit_start = time.time()
    run_compiled_trial(compiled_run, rng_key, jax_init)
    jit_time = time.time() - jit_start
    print(f"NumPyro: JIT warmup done ({jit_time:.2f}s)", flush=True)

    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        dump_mcmc_hlo(compiled_run, rng_key, jax_init, module_path.replace(".", "_"))

    # First real trial
    t0 = time.time()
    samples = run_compiled_trial(compiled_run, rng_key, jax_init)
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
            # Per-trial init: a model that supplies generate_init_params sends a
            # fresh (seed-dependent) init per trial; otherwise, under --matched-init
            # reuse the fixed deterministic matched init (`jax_init`) for every
            # trial (identical start across trials, matching the Impulse/Stan/Turing
            # arms). Only when neither applies do we fall back to NumPyro's default.
            new_raw = req.get("init_params")
            if matched:
                # Matched init wins over any per-trial driver init: reuse the
                # fixed deterministic matched init (`jax_init`) for EVERY trial.
                new_init = jax_init
            elif new_raw is not None:
                new_init = prep_init(new_raw)
            else:
                new_init = None

            new_rng = make_rng_key(new_seed)
            profile_dir = req.get("profile")

            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
                print(f"Profiling -> {profile_dir}", flush=True)
                with jax.profiler.trace(profile_dir, create_perfetto_link=False):
                    t0 = time.time()
                    new_samples = run_compiled_trial(compiled_run, new_rng, new_init)
                    new_run_time = time.time() - t0
            else:
                t0 = time.time()
                new_samples = run_compiled_trial(compiled_run, new_rng, new_init)
                new_run_time = time.time() - t0

            write_output(new_output_path, new_samples, impl, 0.0, new_run_time)
            print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
