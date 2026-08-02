# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

from jsonargparse import ActionJsonSchema, ArgumentParser, dict_to_namespace

from .lib import model_helper, ppl_helper, reports, utils


# The following schema defines the experiment that PPLBench should run.
SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "args": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 2},
                        "k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["n"],
                    "additionalProperties": True,
                    "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                },
                "seed": {"type": "integer", "minimum": 1},
                "package": {"type": "string"},  # defaults to pplbench.models
            },
            "required": ["class"],
            "additionalProperties": False,
        },
        "iterations": {"type": "integer", "minimum": 1},
        "num_warmup": {"type": "integer", "minimum": 0},
        "trials": {"type": "integer", "minimum": 2},
        "profile": {"type": "boolean"},
        "profile_run": {"type": "boolean"},
        "num_profiled": {"type": "integer"},
        "strip_profiled_names": {"type": "boolean"},
        "profiling_tools_dir": {"type": "string"},
        "profiling_type": {"type": "string", "enum": ["deterministic", "statistical"]},
        "ppls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "inference": {
                        "type": "object",
                        "properties": {
                            "class": {"type": "string"},
                            "num_warmup": {"type": "integer", "minimum": 0},
                            "compile_args": {
                                "type": "object",
                                "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                                "additionalProperties": True,
                            },
                            "infer_args": {
                                "type": "object",
                                "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                                "additionalProperties": True,
                            },
                        },
                        "required": ["class"],
                        "additionalProperties": False,
                    },
                    "legend": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "color": {"type": "string"},
                        },
                    },
                    "seed": {"type": "integer", "minimum": 1},
                    "trials": {"type": "integer", "minimum": 1},
                    # package defaults to pplbench.ppls.<ppl name>
                    "package": {"type": "string"},
                    "optional": {"type": "boolean"},
                },
                "required": ["name", "inference"],
                "additionalProperties": False,
            },
        },
        "loglevels": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
        },
        "save_samples": {"type": "boolean"},  # default "false"
        "output_root_dir": {"type": "string"},  # defaults to "./outputs"
        "figures": {
            "type": "object",
            "properties": {
                "generate_pll": {"type": "boolean"},  # default "true"
                "suffix": {"type": "string"},  # default "png"
            },
            "additionalProperties": False,
        },
    },
    "required": ["model", "ppls", "iterations", "trials"],
    "additionalProperties": False,
}

LOGGER = logging.getLogger("pplbench")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    validate = "--validate" in args
    if validate:
        args = [a for a in args if a != "--validate"]

    ppls_filter = None
    include_optional = False
    if "--ppls-all" in args:
        idx = args.index("--ppls-all")
        ppls_filter = args[idx + 1].split(",")
        include_optional = True
        args = args[:idx] + args[idx + 2:]
    elif "--ppls" in args:
        idx = args.index("--ppls")
        ppls_filter = args[idx + 1].split(",")
        args = args[:idx] + args[idx + 2:]

    config = read_config(args)

    # Strip optional entries (e.g. "Impulse (no opt)") unless --ppls-all
    if not include_optional:
        config.ppls = [
            p for p in config.ppls
            if not getattr(p, "optional", False)
        ]

    if ppls_filter:
        _apply_ppls_filter(config, ppls_filter)

    if validate:
        _apply_validate_overrides(config)

    # Propagate PPLBENCH_PROFILE env var into config so ppl_helper
    # profiles trial 1 in server mode (in addition to the initial run
    # which harness.jl/numpyro_harness.py handle via the env var directly).
    if os.environ.get("PPLBENCH_PROFILE"):
        config.profile = True

    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        config.trials = 1
        for ppl in config.ppls:
            if hasattr(ppl, "trials"):
                ppl.trials = 1

    output_dir = utils.create_output_dir(config)
    os.environ["PPLBENCH_OUTPUT_DIR"] = output_dir
    avail = len(os.sched_getaffinity(0))
    total = os.cpu_count()
    print(f"CPUs: {avail} available / {total} total")
    print(f"Output dir: {output_dir}")
    if validate:
        print("Validate mode: NumPyro vs Impulse pointwise agreement "
              "(NUTS: 5 samples, no warmup, adaptation off; SVI: fitted guide loc/scale + ELBO)")
    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        print(f"MLIR dump: {os.path.join(output_dir, 'sicm_dumps')}")
    if os.environ.get("PPLBENCH_PROFILE"):
        print(f"XLA profile: {os.path.join(output_dir, 'traces')}")
    configure_logging(config, output_dir)

    if hasattr(config, "profile_run") and config.profile_run:
        from .lib.ppl_profiler import PPL_Profiler
        ppl_profiler = PPL_Profiler(output_dir, config)

    model_cls = model_helper.find_model_class(config.model)
    all_ppl_details = ppl_helper.find_ppl_details(config)
    train_data, test_data = model_helper.simulate_data(config.model, model_cls)
    (
        all_variable_metrics_data,
        all_other_metrics_data,
    ) = ppl_helper.collect_samples_and_stats(config, model_cls, all_ppl_details, train_data, test_data, output_dir)

    if validate:
        _run_validation(output_dir, config, all_ppl_details)
    else:
        reports.generate_plots(
            output_dir, config, all_ppl_details,
            all_variable_metrics_data, all_other_metrics_data,
        )

    # Append per-PPL summary to results CSV (append-only, never overwrites)
    _append_results_csv(config, all_other_metrics_data, output_dir)

    if hasattr(config, "profile_run") and config.profile_run:
        ppl_profiler.finish_profiling()

    LOGGER.info(f"Output saved in '{output_dir}'")


def _append_results_csv(config, metrics_data, output_dir):
    """Append one row per PPL to results.csv (never overwrites)."""
    import numpy as np

    csv_path = os.path.join(os.path.dirname(output_dir), "results.csv")
    fields = [
        "timestamp", "output_dir", "model", "model_args",
        "num_warmup", "num_samples",
        "ppl", "legend",
        "compile_time_s", "mean_runtime_s", "all_runtimes_s",
        "neff_min", "neff_median", "neff_max",
        "neff_per_s_min", "neff_per_s_median", "neff_per_s_max",
        "stuck",
    ]

    write_header = not os.path.exists(csv_path)
    model_name = getattr(config.model, "class", "unknown")
    model_args = str(config.model.args.__dict__) if hasattr(config.model, "args") else ""
    num_warmup = getattr(config, "num_warmup", 0)
    num_samples = config.iterations - num_warmup
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ppl_names = list(metrics_data["ppl"].values)
    timing = metrics_data["timing"].values        # [ppl, chain, phase(compile/run)]
    neff = metrics_data["overall_neff"].values     # [ppl, percentile(min/med/max)]
    neff_ps = metrics_data["overall_neff_per_time"].values

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fields)

        for i, ppl_name in enumerate(ppl_names):
            # Find legend name from config
            legend = ppl_name
            for p in config.ppls:
                if p.name == ppl_name and hasattr(p, "legend"):
                    legend = p.legend.name if hasattr(p.legend, "name") else str(p.legend)
                    break

            run_times = timing[i, :, 1]  # phase 1 = run time
            valid_times = run_times[~np.isnan(run_times)]
            compile_time = timing[i, 0, 0]  # first trial compile time
            # Steady-state = exclude trial 0
            steady = valid_times[1:] if len(valid_times) > 1 else valid_times
            mean_runtime = float(np.mean(steady)) if len(steady) > 0 else float('nan')

            # Detect stuck chains: if n_eff equals num_samples for all variables,
            # the chain is likely stuck (all samples identical → trivial ESS)
            neff_vals = neff[i]
            stuck = bool(np.all(np.isnan(neff_vals)) or
                         (neff_vals[0] == neff_vals[2] and neff_vals[0] == num_samples))

            writer.writerow([
                timestamp,
                os.path.basename(output_dir),
                model_name,
                model_args,
                num_warmup,
                num_samples,
                ppl_name,
                legend,
                f"{compile_time:.2f}",
                f"{mean_runtime:.3f}",
                ";".join(f"{t:.3f}" for t in valid_times),
                f"{neff_vals[0]:.1f}",
                f"{neff_vals[1]:.1f}",
                f"{neff_vals[2]:.1f}",
                f"{neff_ps[i][0]:.1f}",
                f"{neff_ps[i][1]:.1f}",
                f"{neff_ps[i][2]:.1f}",
                stuck,
            ])

    LOGGER.info(f"Results appended to {csv_path}")


def _apply_ppls_filter(config, ppls_filter):
    """Keep only PPL entries whose name matches one of the filter values."""
    kept = [ppl for ppl in config.ppls if ppl.name in ppls_filter]
    if not kept:
        raise RuntimeError(
            f"--ppls filter {ppls_filter} matched none of the config's PPLs: "
            f"{[p.name for p in config.ppls]}"
        )
    config.ppls = kept


# Tolerances for the validate comparison. Do NOT loosen these to force a pass:
# a genuine numerical divergence is a finding to report, not a tuning knob.
_VALIDATE_ATOL = 1e-4
_VALIDATE_RTOL = 1e-3

# Per-model step size for --validate ONLY (never for benchmarking).
#
# WHY THIS EXISTS. The benchmark configs all run NUTS at step_size 0.1 WITH step
# size adaptation, which immediately shrinks it. Validate mode must keep
# adaptation OFF to stay bitwise comparable, so it runs at the raw config value
# forever. For the three large-n models that value is far above the leapfrog
# stability limit at the matched-init point, so the very first leapfrog step
# blows the Hamiltonian past NUTS's max_delta_energy, the base tree is marked
# diverging, and the biased transition kernel zeroes the transition probability
# (numpyro hmc_util.py `_biased_transition_kernel`; Impulse
# HMCUtils.cpp `computeBiasedTransitionProb`). The proposal therefore never
# leaves the initial point and the chain is FROZEN: the gate then compares a
# constant against a constant and cannot detect layout / RNG / gradient bugs.
#
# The stability limit eps_crit = 2/sqrt(lambda_max(Hessian of U at the matched
# init)) measured on these exact configs:
#
#   model                     lambda_max   eps_crit   0.1/eps_crit   frozen?
#   logistic_regression         5.56e6     8.48e-4        118x        yes
#   robust_regression           1.68e4     1.54e-2       6.49x        yes
#   n_schools                   9.17e3     2.09e-2       4.79x        yes
#   crowd_sourced_annotation      82.2     2.21e-1       0.45x        no
#
# Each value below is the LARGEST step size at which a 5-draw adapt-off NUTS run
# from the matched init has zero divergences and accept_prob >= 0.9 on every
# draw of both validate trial seeds (1396133 and 2993578). Models absent from
# this table already move at their config step size and are left untouched.
#
# This applies ONLY in validate mode and reaches BOTH arms through the same
# `infer_args.step_size` that becomes `--step-size` on the numpyro and impulse
# subprocess command lines. The benchmark configs in pplbench_configs/standard/
# keep step_size 0.1 and must not be edited for this.
_VALIDATE_STEP_SIZE = {
    "logistic_regression.LogisticRegression": 0.001,
    "robust_regression.RobustRegression": 0.005,
    "n_schools.NSchools": 0.01,
}


def _inference_kind(ppl) -> str:
    """Trailing component of a ppl's inference `class`, e.g. 'MCMC' or 'SVI'."""
    return getattr(ppl.inference, "class").rsplit(".", 1)[-1]


def _apply_validate_overrides(config):
    """Make ``--validate`` a self-contained correctness mode.

    Keeps only the numpyro and impulse arms (first occurrence of each), then
    DISPATCHES on their inference class:

      * MCMC (NUTS) -> ``_apply_validate_overrides_mcmc`` (implemented here).
      * SVI         -> ``NotImplementedError``; the variational correctness
                       protocol is owned by the separate SVI validate agent
                       (see the EXTENSION POINT below).

    This dispatch keeps the NUTS protocol from silently running against a
    variational arm once SVI validate configs exist.
    """
    kept = []
    seen = set()
    for ppl in config.ppls:
        if ppl.name in ("numpyro", "impulse") and ppl.name not in seen:
            seen.add(ppl.name)
            kept.append(ppl)
    if len(kept) < 2:
        raise RuntimeError(
            "Validate mode requires both 'numpyro' and 'impulse' entries in config"
        )

    kinds = {_inference_kind(ppl) for ppl in kept}
    if kinds == {"MCMC"}:
        _apply_validate_overrides_mcmc(config, kept)
    elif kinds <= {"SVI"}:
        # === EXTENSION POINT: SVI validate ===
        # Variational correctness protocol: pin a short deterministic optimization
        # from a matched guide init (loc = 0) with a matched reparam-noise stream,
        # paired with `_run_validation_svi` (dispatched from `_run_validation`).
        _apply_validate_overrides_svi(config, kept)
    else:
        raise RuntimeError(
            f"Validate mode: kept arms have mixed/unknown inference kinds {kinds}; "
            "expected all MCMC or all SVI"
        )


def _apply_validate_overrides_mcmc(config, kept):
    """NUTS correctness protocol.

    Forces a short, non-adaptive run from a matched starting point so NumPyro and
    Impulse are pointwise comparable: 5 samples, no warmup, adaptation off, 2
    trials, and matched initialization on every kept arm.

    Adaptation defaults OFF, and step-size adaptation in particular cannot be
    part of a *bitwise* cross-framework check (this is fundamental, not the since
    resolved dual-averaging init drift). Isolating the two adaptations on
    logistic_regression at 200 warmup steps: mass-matrix adaptation alone stays
    pointwise (alpha max_diff 2.0e-5; robust_regression 5.1e-11), but step-size
    adaptation alone DIVERGES (alpha 8.4). Dual averaging's log/exp/pow carries an
    unavoidable ~1e-13 XLA-vs-JAX transcendental ULP floor; NUTS is chaotic, so
    for a curved posterior that seed amplifies exponentially (full-adapt alpha
    max_diff vs warmup: 0->0.0, 1->0.0, 5->5.7e-5, 20->2.8e-3, 200->6.2), while a
    mild posterior does not saturate (the Gaussian nuts_adaptation_combinations
    test holds to ~1e-8 at 200 warmup). It is a ULP floor, not a formula bug: the
    per-step difference is ULP-scale (mass-only and the Gaussian stay tight).

    The env knobs below expose the stronger mass-matrix-adaptation check (which
    exercises Welford covariance + Cholesky + momentum resampling and still
    passes) without shipping it as the bitwise default:
      PPLBENCH_VALIDATE_ADAPT_MASS=1   mass-matrix adaptation on (recommended
                                       stronger check; pointwise within ~1e-5)
      PPLBENCH_VALIDATE_ADAPT_STEP=1   step-size adaptation on (diverges on
                                       chaotic models; for investigation only)
      PPLBENCH_VALIDATE_WARMUP=N       warmup steps (default 0; needs >0 for any
                                       adaptation to run)
    """
    _warmup = int(os.environ.get("PPLBENCH_VALIDATE_WARMUP", "0"))
    _ass = os.environ.get("PPLBENCH_VALIDATE_ADAPT_STEP", "0") == "1"
    _amm = os.environ.get("PPLBENCH_VALIDATE_ADAPT_MASS", "0") == "1"
    config.iterations = _warmup + 5
    config.num_warmup = _warmup
    config.trials = 2
    # Non-degeneracy: an adapt-off chain run at the benchmark step size is frozen
    # on the large-n models (see _VALIDATE_STEP_SIZE). Applied to every kept arm
    # so both sides run the identical protocol.
    _model_class = getattr(config.model, "class", None)
    _step_size = _VALIDATE_STEP_SIZE.get(_model_class)
    for ppl in kept:
        # A per-ppl `trials` overrides config.trials in ppl_helper, so it must be
        # set here too; otherwise validate would silently run the config's full
        # trial count (e.g. 4) instead of the intended small, well-defined run.
        ppl.trials = 2
        infer = ppl.inference
        if not hasattr(infer, "infer_args"):
            infer.infer_args = SimpleNamespace()
        # Default fixed leapfrog (both flags off): short deterministic trajectory
        # so no ULP floor amplifies. Env knobs above opt into adaptation.
        infer.infer_args.adapt_step_size = _ass
        infer.infer_args.adapt_mass_matrix = _amm
        # Matched init: NumPyro starts at the deterministic canonical constrained
        # point (real=0, positive=1, simplex=uniform) via --matched-init; the
        # impulse arm already pins that same point through its build_constraint,
        # so both arms start identically and every sample is comparable. The
        # impulse harness ignores this kwarg; only the numpyro arm consumes it.
        infer.infer_args.matched_init = True
        if _step_size is not None:
            infer.infer_args.step_size = _step_size
    config.ppls = kept
    if _step_size is not None:
        print(f"Validate step-size override for {_model_class}: "
              f"step_size={_step_size} on arms {[p.name for p in kept]} "
              f"(benchmark configs keep their own value)")


def _apply_validate_overrides_svi(config, kept):
    """SVI (variational) correctness protocol.

    Pins a short, deterministic optimization so NumPyro and Impulse are pointwise
    comparable on the FIT (fitted guide loc/scale + per-step ELBO), NOT the
    post-fit posterior draws (those use a separate draw RNG and legitimately
    differ). Both arms already cold-start the AutoNormal guide at unconstrained
    loc = 0 with init_scale = 0.1 (harness.jl `run_svi` initial_position = zeros;
    numpyro_harness `_svi_init_constrained`), so the guides begin identically;
    the only per-trial variation is the fit RNG seed, which ppl_helper shares
    across arms. `matched_init=True` additionally switches the numpyro arm's fit
    to the rbg key stream that matches Impulse's reparam noise (the impulse arm
    ignores this kwarg).
    """
    # A short deterministic optimization is enough to expose any divergence while
    # staying well inside the ULP regime the unit tests confirm (test/probprog/
    # vi_bayesian_logreg.jl: ELBO/loc identical to ~1e-15 over 5 steps).
    config.iterations = 5       # K = post-fit guide draws (NOT compared for SVI)
    config.num_warmup = 0
    config.trials = 2
    for ppl in kept:
        # A per-ppl `trials` overrides config.trials in ppl_helper, so pin it here
        # too (mirrors the MCMC fix) instead of silently using the config's 4.
        ppl.trials = 2
        infer = ppl.inference
        if not hasattr(infer, "infer_args"):
            infer.infer_args = SimpleNamespace()
        # Short, matched optimization. step_size / init_scale / guide / optimizer
        # already come from the svi/*.json configs; set them explicitly so the run
        # is deterministic regardless of the config's values.
        infer.infer_args.num_steps = 20
        infer.infer_args.num_particles = 1
        infer.infer_args.guide = "AutoNormal"
        infer.infer_args.optimizer = "Adam"
        infer.infer_args.step_size = 0.01
        infer.infer_args.init_scale = 0.1
        # Matched reparam-noise stream: only the numpyro SVI arm consumes this
        # (switches to the rbg key from make_rng_key); the impulse arm ignores it
        # since its enzyme.random DEFAULT already IS that rbg stream.
        infer.infer_args.matched_init = True
    config.ppls = kept


def _run_validation(output_dir, config, all_ppl_details):
    """Compare the two kept arms' sample outputs. Dispatches on inference kind."""
    kinds = {p.inference_class.__name__ for p in all_ppl_details}
    if kinds == {"MCMC"}:
        _run_validation_mcmc(output_dir, config, all_ppl_details)
    elif kinds <= {"SVI"}:
        # === EXTENSION POINT: SVI validate ===
        # Compare the fitted guide (loc/scale + per-step ELBO), NOT posterior draws.
        _run_validation_svi(output_dir, config, all_ppl_details)
    else:
        raise RuntimeError(
            f"Validate mode: kept arms have mixed/unknown inference kinds {kinds}; "
            "expected all MCMC or all SVI"
        )


def _run_validation_mcmc(output_dir, config, all_ppl_details):
    """NUTS correctness comparison: NumPyro vs Impulse sample outputs.

    Prints per-variable max_diff + PASS/FAIL, writes ``validation_summary.json``
    into ``output_dir``, and exits nonzero if any variable differs beyond
    tolerance.

    NON-DEGENERACY. Agreement is only evidence of correctness if the chains
    actually moved: a frozen chain compares a constant against a constant and
    passes no matter how broken the sampler is. So alongside max_diff we also
    record, per variable per arm, the standard deviation over the draw axis
    (reduced with max and min over chains and components). A variable whose
    max std over draws is exactly 0 on BOTH arms is reported as DEGENERATE and
    fails the gate under its own clearly labelled condition. The tolerance
    semantics above are untouched by this: _VALIDATE_ATOL / _VALIDATE_RTOL and
    the ``all_close`` verdict mean exactly what they meant before.
    """
    import glob
    import json
    import numpy as np

    sample_files = sorted(glob.glob(os.path.join(output_dir, "samples*.nc")))
    if not sample_files:
        LOGGER.warning("No sample files found for validation comparison")
        return

    import xarray as xr
    ds = xr.open_dataset(sample_files[0])

    names = [p.name for p in all_ppl_details]
    if len(names) < 2:
        LOGGER.warning("Need at least 2 PPLs for validation")
        return

    ppl_a, ppl_b = names[0], names[1]
    # infer_args of the first arm carry the shared step size / tree depth.
    infer_a = all_ppl_details[0].infer_args or {}
    print(f"\n=== Validation: {ppl_a} vs {ppl_b} ===")
    all_close = True
    variables = {}
    degenerate_vars = []

    def _movement(values, dims):
        """(max, min) std over the draw axis, reduced over chains/components.

        `dims` are the variable's dims INCLUDING the leading `ppl` dim, which
        `.sel(ppl=...)` has already dropped from `values`.
        """
        if "draw" not in dims:
            return float("nan"), float("nan")
        axis = list(dims).index("draw") - 1
        s = np.nanstd(values, axis=axis)
        return float(np.nanmax(s)), float(np.nanmin(s))

    for var in ds.data_vars:
        a = ds[var].sel(ppl=ppl_a).values
        b = ds[var].sel(ppl=ppl_b).values
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        close = bool(np.allclose(a, b, atol=_VALIDATE_ATOL, rtol=_VALIDATE_RTOL))
        max_diff = float(np.nanmax(np.abs(a - b)))
        exactly_zero = bool(max_diff == 0.0)
        status = "PASS" if close else "FAIL"
        tag = "exact" if exactly_zero else ("within-tol" if close else "DIVERGES")
        # Movement evidence: did the chain go anywhere at all?
        std_a, min_std_a = _movement(a, ds[var].dims)
        std_b, min_std_b = _movement(b, ds[var].dims)
        moved = bool(std_a > 0.0 and std_b > 0.0)
        degenerate = bool(std_a == 0.0 and std_b == 0.0)
        if degenerate:
            degenerate_vars.append(str(var))
        marker = "  <<< DEGENERATE: chain FROZEN on both arms" if degenerate else ""
        print(f"  {var}: {status} (max_diff={max_diff:.6e}) [{tag}] "
              f"std_over_draws: {ppl_a}={std_a:.6e} {ppl_b}={std_b:.6e}{marker}")
        variables[str(var)] = {
            "max_diff": max_diff,
            "pass": close,
            "exactly_zero": exactly_zero,
            "moved": moved,
            "degenerate": degenerate,
            "std_over_draws": {ppl_a: std_a, ppl_b: std_b},
            "min_std_over_draws": {ppl_a: min_std_a, ppl_b: min_std_b},
        }
        if not close:
            all_close = False

    summary = {
        "model": getattr(config.model, "class", "unknown"),
        "model_args": (config.model.args.__dict__
                       if hasattr(config.model, "args") else {}),
        "seed": getattr(config.model, "seed", None),
        "iterations": config.iterations,
        "num_warmup": getattr(config, "num_warmup", 0),
        "trials": config.trials,
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "step_size": infer_a.get("step_size"),
        "max_tree_depth": infer_a.get("max_tree_depth"),
        "atol": _VALIDATE_ATOL,
        "rtol": _VALIDATE_RTOL,
        "all_close": all_close,
        "any_degenerate": bool(degenerate_vars),
        "degenerate_variables": degenerate_vars,
        "variables": variables,
    }
    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Validation summary written to {summary_path}")

    if degenerate_vars:
        print("")
        print("!!! DEGENERATE GATE: the chain never moved for "
              f"{', '.join(degenerate_vars)} (std over draws is exactly 0 on "
              f"both {ppl_a} and {ppl_b}, every draw and every chain).")
        print("!!! Agreement here is meaningless: the gate compared a constant "
              "against a constant and would pass with a completely broken "
              "sampler. Fix the protocol (see _VALIDATE_STEP_SIZE); do NOT "
              "loosen the tolerances and do NOT ignore this.")

    if not all_close:
        print("Validation FAILED: some variables differ beyond tolerance")
        sys.exit(1)
    if degenerate_vars:
        print("Validation FAILED (degenerate): variables matched within "
              "tolerance but the chain is frozen, so the gate proved nothing")
        sys.exit(2)
    print("Validation PASSED: all variables match within tolerance "
          "and every variable moved")


def _run_validation_svi(output_dir, config, all_ppl_details):
    """SVI correctness comparison: NumPyro vs Impulse fitted guide.

    Compares the FIT that is byte-identical across the two backends: per-site
    fitted guide ``loc``/``scale`` (unconstrained space) and the per-step ELBO
    trajectory. The post-fit posterior draws in samples*.nc are intentionally NOT
    compared: they use a separate draw RNG and legitimately differ.

    Both harnesses write ``svi_diagnostics/<name>_seed<seed>.json`` per trial
    (numpyro files carry a ``numpyro_`` prefix; impulse files use the display
    model name). We pair the two by trial seed, compare each quantity pointwise at
    the same tolerance as the MCMC path, print per-site max_diff + PASS/FAIL, write
    ``validation_summary.json`` and exit nonzero on a genuine divergence.
    """
    import glob
    import json
    import re
    import numpy as np

    diag_dir = os.path.join(output_dir, "svi_diagnostics")
    files = glob.glob(os.path.join(diag_dir, "*_seed*.json"))
    if not files:
        LOGGER.warning("No svi_diagnostics files found for validation comparison")
        return

    def _seed_of(path):
        m = re.search(r"_seed(-?\d+)\.json$", os.path.basename(path))
        return int(m.group(1)) if m else None

    numpyro_by_seed, impulse_by_seed = {}, {}
    for p in files:
        s = _seed_of(p)
        if s is None:
            continue
        if os.path.basename(p).startswith("numpyro_"):
            numpyro_by_seed[s] = p
        else:
            impulse_by_seed[s] = p

    shared = sorted(set(numpyro_by_seed) & set(impulse_by_seed))
    if not shared:
        LOGGER.warning(
            "No shared seeds between numpyro/impulse svi_diagnostics "
            f"(numpyro={sorted(numpyro_by_seed)}, impulse={sorted(impulse_by_seed)})"
        )
        return

    def _cmp(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        close = bool(np.allclose(a, b, atol=_VALIDATE_ATOL, rtol=_VALIDATE_RTOL))
        max_diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        exact = bool(max_diff == 0.0)
        tag = "exact" if exact else ("within-tol" if close else "DIVERGES")
        return close, max_diff, exact, tag

    ppl_a, ppl_b = "numpyro", "impulse"
    print(f"\n=== SVI Validation: {ppl_a} vs {ppl_b} (fitted guide) ===")
    all_close = True
    per_seed = {}
    for seed in shared:
        with open(numpyro_by_seed[seed]) as f:
            nd = json.load(f)
        with open(impulse_by_seed[seed]) as f:
            imd = json.load(f)
        print(f"\n-- seed {seed} --")
        entry = {"loc": {}, "scale": {}, "elbo": None}

        # Fitted guide loc / scale, per site, keyed by NAME (order-independent).
        for quantity in ("loc", "scale"):
            na = nd.get(quantity, {}) or {}
            ia = imd.get(quantity, {}) or {}
            sites = sorted(set(na) & set(ia))
            only_one = sorted(set(na) ^ set(ia))
            if only_one:
                print(f"  {quantity}: SITE MISMATCH, only-in-one={only_one}")
                all_close = False
            for site in sites:
                close, max_diff, exact, tag = _cmp(na[site], ia[site])
                status = "PASS" if close else "FAIL"
                print(f"  {quantity}[{site}]: {status} "
                      f"(max_diff={max_diff:.6e}) [{tag}]")
                entry[quantity][site] = {
                    "max_diff": max_diff, "pass": close, "exactly_zero": exact,
                }
                if not close:
                    all_close = False

        # ELBO trajectory, with defensive constant-offset handling. loc/scale is
        # the verdict; a pure per-step constant ELBO offset is reported but does
        # NOT fail the run (the confirmed unit test shows no offset in practice).
        ne = np.asarray(nd.get("elbo_trajectory", []), dtype=float)
        ie = np.asarray(imd.get("elbo_trajectory", []), dtype=float)
        if ne.size and ie.size and ne.size == ie.size:
            close, max_diff, exact, tag = _cmp(ne, ie)
            offset = None
            if not close:
                diff = ne - ie
                off = float(np.median(diff))
                if bool(np.allclose(diff, off,
                                    atol=_VALIDATE_ATOL, rtol=_VALIDATE_RTOL)):
                    offset = off
                    tag = "constant-offset"
            status = "PASS" if (close or offset is not None) else "FAIL"
            extra = "" if offset is None else f" offset={offset:.6e}"
            print(f"  elbo: {status} (max_diff={max_diff:.6e}) [{tag}]{extra}")
            entry["elbo"] = {
                "max_diff": max_diff, "pass": close, "exactly_zero": exact,
                "constant_offset": offset,
            }
            if not close and offset is None:
                all_close = False
        else:
            print(f"  elbo: SKIP (numpyro n={ne.size}, impulse n={ie.size})")
        per_seed[str(seed)] = entry

    infer_a = all_ppl_details[0].infer_args or {}
    summary = {
        "model": getattr(config.model, "class", "unknown"),
        "model_args": (config.model.args.__dict__
                       if hasattr(config.model, "args") else {}),
        "seed": getattr(config.model, "seed", None),
        "inference": "SVI",
        "num_steps": infer_a.get("num_steps"),
        "step_size": infer_a.get("step_size"),
        "init_scale": infer_a.get("init_scale"),
        "trials": config.trials,
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "atol": _VALIDATE_ATOL,
        "rtol": _VALIDATE_RTOL,
        "shared_seeds": shared,
        "all_close": all_close,
        "per_seed": per_seed,
    }
    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nValidation summary written to {summary_path}")

    if all_close:
        print("SVI Validation PASSED: fitted guide matches within tolerance")
    else:
        print("SVI Validation FAILED: fitted guide differs beyond tolerance")
        sys.exit(1)


def read_config(args: Optional[List[str]]) -> SimpleNamespace:
    """
    Parse command line arguments and return a JSON object.
    :returns: benchmark configuration.
    """
    parser = ArgumentParser()
    parser.add_argument("config", action=ActionJsonSchema(schema=SCHEMA), help="%s")
    config = parser.parse_args(args).config
    with mock.patch("jsonargparse.namespace.Namespace", SimpleNamespace):
        config = dict_to_namespace(config)

    # default num_warmup to half of num_sample
    if not hasattr(config, "num_warmup"):
        config.num_warmup = config.iterations // 2

    return config


def configure_logging(config: SimpleNamespace, output_dir: str) -> None:
    """
    Configure logging based on `config.loglevel` and add a stream handler.
    :param config: benchmark configuration
    :output_dir: directory to save the output
    """
    # set log level to INFO by default on root logger
    logging.getLogger().setLevel("INFO")
    # setup logging for all other requested loglevels
    if hasattr(config, "loglevels"):
        for key, val in config.loglevels.__dict__.items():
            logging.getLogger(key).setLevel(getattr(logging, val))
    # create a handler at the root level to display to stdout
    # and another to write to a log file
    for ch in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_dir, "logging.txt"), encoding="utf-8"),
    ]:
        formatter = logging.Formatter(LOG_FORMAT)
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
    LOGGER.debug(f"config - {str(config)}")
