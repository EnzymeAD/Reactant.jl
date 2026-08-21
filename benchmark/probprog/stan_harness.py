#!/usr/bin/env python3
"""
Stan/CmdStanPy subprocess server for PPLBench.
Mirrors the Impulse harness.jl protocol: ###READY###/###DONE###/EXIT.
"""
import argparse
import importlib
import json
import os
import re
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
              init_params, num_chains=1):
    inits = None
    if init_params is not None:
        inits = {}
        for k, v in init_params.items():
            arr = np.array(v)
            inits[k] = arr.item() if arr.ndim == 1 and arr.shape[0] == 1 else arr

    fit = model.sample(
        data=stan_data,
        chains=num_chains,
        parallel_chains=num_chains,
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


# =============================================================================
# SVI support (Stan ADVI, mean-field)
# =============================================================================

def _format_inits(init_params):
    """Constrained inits dict -> CmdStan `inits` format (scalars unwrapped)."""
    if init_params is None:
        return None
    inits = {}
    for k, v in init_params.items():
        arr = np.array(v)
        inits[k] = arr.item() if arr.ndim == 1 and arr.shape[0] == 1 else arr
    return inits


_ADVI_NOT_CONVERGED = "The algorithm may not have converged."


def _advi_transcript_diagnostics(vb):
    """Read CmdStan's ADVI console transcript for the convergence verdict.

    CmdStan's ADVI (stan/variational/advi.hpp stochastic_gradient_ascent) stops
    for one of two reasons and says which one on the console:
      * the rolling relative-ELBO test fired -> "MEAN ELBO CONVERGED" and/or
        "MEDIAN ELBO CONVERGED";
      * the iteration cap was hit -> "The maximum number of iterations is
        reached! The algorithm may not have converged."
    cmdstanpy only regex-matches the second string (model.py:1565-1582) and
    either raises or warns; nothing machine-readable is kept. We record the
    verdict and the last ELBO row per trial so the arm's convergence status is
    an artifact of the run, not just a log line."""
    out = {
        "converged": None,
        "stop_reason": None,
        "final_iter": None,
        "final_elbo": None,
        "delta_elbo_mean": None,
        "delta_elbo_med": None,
        "eta": None,
    }
    try:
        path = vb.runset.stdout_files[0]
        with open(path) as f:
            text = f.read()
    except Exception as e:
        out["stop_reason"] = f"transcript unavailable: {e}"
        return out

    hit_cap = _ADVI_NOT_CONVERGED in text
    mean_conv = "MEAN ELBO CONVERGED" in text
    med_conv = "MEDIAN ELBO CONVERGED" in text
    out["converged"] = not hit_cap
    reasons = []
    if mean_conv:
        reasons.append("MEAN ELBO CONVERGED")
    if med_conv:
        reasons.append("MEDIAN ELBO CONVERGED")
    if hit_cap:
        reasons.append("max iterations reached")
    out["stop_reason"] = "; ".join(reasons) if reasons else "unknown"
    out["elbo_previous_best_warning"] = (
        "The ELBO at a previous iteration is larger" in text
    )

    # Last "  <iter>  <elbo>  <delta_mean>  <delta_med>[  notes]" row.
    row = re.compile(
        r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.M
    )
    rows = row.findall(text)
    if rows:
        it, elbo, dmean, dmed = rows[-1]
        out["final_iter"] = int(it)
        out["final_elbo"] = float(elbo)
        out["delta_elbo_mean"] = float(dmean)
        out["delta_elbo_med"] = float(dmed)
        out["elbo_trace_tail"] = [
            {"iter": int(a), "elbo": float(b), "delta_mean": float(c),
             "delta_med": float(d)}
            for a, b, c, d in rows[-10:]
        ]
    eta = re.search(r"^eta\s*=\s*([0-9.eE+-]+)", text, re.M)
    if eta:
        out["eta"] = float(eta.group(1))
    return out


def run_vi_trial(model, stan_data, seed, output_samples, grad_samples,
                 vi_algorithm, vi_iter, require_converged, init_params=None,
                 advi_opts=None, show_console=False):
    """Run one Stan ADVI (variational) fit and return per-draw samples.

    Only this call is timed as the inference phase by the caller (compile is
    timed separately at CmdStanModel construction). Stan ADVI runs its OWN
    adaptive stochastic optimizer (eta adaptation + RMSProp-style updates with a
    relative-ELBO convergence test); it cannot be pinned to Adam lr=0.01 / 2000
    steps like the NumPyro / Impulse / Turing arms, so it uses its own defaults
    for eta/iter/tol_rel_obj. `grad_samples=1` mirrors the one-particle ELBO
    gradient; `draws=output_samples` gives the K post-fit posterior draws.
    `init_params` (matched-init control) is the deterministic constrained
    starting point for the variational optimizer (CmdStan `inits`).

    `advi_opts` carries any explicitly-requested CmdStan ADVI settings
    (elbo_samples / tol_rel_obj / eval_elbo / eta / adapt_engaged / adapt_iter).
    Keys absent from the dict are NOT passed to CmdStan, so they keep CmdStan's
    own defaults. Returns `(samples, diagnostics)`."""
    kwargs = dict(
        data=stan_data,
        seed=seed,
        algorithm=vi_algorithm,       # 'meanfield' == AutoNormal mean-field
        grad_samples=grad_samples,    # 1 MC sample per gradient
        draws=output_samples,         # K posterior draws from the fitted guide
        require_converged=require_converged,
        show_console=show_console,
    )
    inits = _format_inits(init_params)
    if inits is not None:
        kwargs["inits"] = inits
    if vi_iter is not None:
        kwargs["iter"] = vi_iter
    if advi_opts:
        kwargs.update(advi_opts)
    vb = model.variational(**kwargs)
    diag = _advi_transcript_diagnostics(vb)
    # mean=False -> per-draw variational sample, leading axis = output_samples.
    return vb.stan_variables(mean=False), diag


def _vi_write_diagnostics(model_name, seed, samples, run_time, num_steps,
                          advi_diag=None, advi_settings=None):
    diag_dir = os.path.join(
        os.environ.get("PPLBENCH_OUTPUT_DIR", "outputs"), "svi_diagnostics"
    )
    try:
        os.makedirs(diag_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(model_name))
        path = os.path.join(diag_dir, f"stan_{safe}_seed{seed}.json")
        # Per-site posterior means over the K draws: the primary
        # converged-vs-buggy diagnostic (compared against the NumPyro reference).
        site_means = {}
        for k, v in samples.items():
            arr = np.asarray(v, dtype=float)
            site_means[k] = np.mean(arr, axis=0).tolist()
        record = {
            "model": str(model_name),
            "seed": int(seed),
            "num_steps": num_steps,
            "run_time_s": float(run_time),
            "site_means": site_means,
        }
        if advi_settings is not None:
            record["advi_settings"] = advi_settings
        if advi_diag is not None:
            record["advi_convergence"] = advi_diag
        with open(path, "w") as f:
            json.dump(record, f)
    except Exception as e:  # diagnostics are best-effort
        print(f"Stan VI: failed to write diagnostics: {e}", flush=True)


def _collect_advi_opts(args):
    """Explicitly-requested CmdStan ADVI settings only.

    Anything the caller did not set is left out so CmdStan keeps its own
    default. Returns (kwargs_for_cmdstan, human_readable_settings)."""
    opts = {}
    if args.vi_elbo_samples is not None:
        opts["elbo_samples"] = args.vi_elbo_samples
    if args.vi_tol_rel_obj is not None:
        opts["tol_rel_obj"] = args.vi_tol_rel_obj
    if args.vi_eval_elbo is not None:
        opts["eval_elbo"] = args.vi_eval_elbo
    if args.vi_eta is not None:
        opts["eta"] = args.vi_eta
    if args.vi_adapt_iter is not None:
        opts["adapt_iter"] = args.vi_adapt_iter
    if args.vi_no_adapt:
        opts["adapt_engaged"] = False
    settings = dict(opts)
    settings["algorithm"] = args.vi_algorithm
    settings["grad_samples"] = args.grad_samples
    settings["iter"] = args.vi_iter
    settings["draws"] = args.output_samples
    return opts, settings


def _report_advi_convergence(seed, diag):
    print(
        f"Stan ADVI [seed {seed}]: converged={diag.get('converged')} "
        f"stop_reason='{diag.get('stop_reason')}' "
        f"final_iter={diag.get('final_iter')} "
        f"final_elbo={diag.get('final_elbo')} "
        f"delta_elbo_mean={diag.get('delta_elbo_mean')} "
        f"delta_elbo_med={diag.get('delta_elbo_med')} "
        f"eta={diag.get('eta')}",
        flush=True,
    )


def vi_main(args, model, stan_data, impl, compile_time, module_path,
            matched_init=None):
    K = args.output_samples
    advi_opts, advi_settings = _collect_advi_opts(args)
    print(f"Stan ADVI: algorithm={args.vi_algorithm}, draws(K)={K}, "
          f"grad_samples={args.grad_samples}, iter={args.vi_iter}, "
          f"require_converged={args.require_converged}, "
          f"matched_init={matched_init is not None}, "
          f"explicit_advi_opts={advi_opts}", flush=True)

    t0 = time.time()
    samples, diag = run_vi_trial(
        model, stan_data, args.seed, K, args.grad_samples,
        args.vi_algorithm, args.vi_iter, args.require_converged, matched_init,
        advi_opts, args.vi_show_console,
    )
    run_time = time.time() - t0
    _report_advi_convergence(args.seed, diag)
    _vi_write_diagnostics(module_path, args.seed, samples, run_time,
                          args.num_steps, diag, advi_settings)
    write_output(args.output, samples, impl, compile_time, run_time, 1, K)
    print(f"First trial: {run_time:.3f}s", flush=True)

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
        t0 = time.time()
        new_samples, new_diag = run_vi_trial(
            model, stan_data, new_seed, K, args.grad_samples,
            args.vi_algorithm, args.vi_iter, args.require_converged,
            matched_init, advi_opts, args.vi_show_console,
        )
        new_run_time = time.time() - t0
        _report_advi_convergence(new_seed, new_diag)
        _vi_write_diagnostics(module_path, new_seed, new_samples, new_run_time,
                              args.num_steps, new_diag, advi_settings)
        write_output(new_output_path, new_samples, impl, 0.0, new_run_time, 1, K)
        print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)


def write_output(output_path, samples, impl, compile_time, run_time,
                 num_chains=1, num_samples=None):
    result = impl.extract_data_from_stan(samples)

    output = {}
    for var in result.data_vars:
        vals = result[var].values
        if num_chains > 1:
            # cmdstanpy pools draws chain-major (all of chain 0, then chain 1,
            # ...); reshape the draw axis to (chain, draw) so the subprocess
            # reader sees a leading chain dim for split-Rhat / pooled-ESS.
            per = vals.shape[0] // num_chains
            vals = vals.reshape((num_chains, per) + vals.shape[1:])
            output[var] = [vals[c].tolist() for c in range(num_chains)]
        elif vals.ndim == 1:
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
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--chain-method", default="parallel")
    parser.add_argument("--method", default="nuts")
    parser.add_argument("--output-samples", type=int, default=200)
    parser.add_argument("--grad-samples", type=int, default=1)
    parser.add_argument("--vi-algorithm", default="meanfield")
    parser.add_argument("--vi-iter", type=int, default=None)
    # CmdStan ADVI knobs. Each defaults to None == "not passed to CmdStan", so
    # CmdStan's own default applies; only explicitly-set knobs deviate.
    parser.add_argument("--vi-elbo-samples", type=int, default=None,
                        help="CmdStan elbo_samples (MC draws per ELBO estimate; "
                             "CmdStan default 100)")
    parser.add_argument("--vi-tol-rel-obj", type=float, default=None,
                        help="CmdStan tol_rel_obj (relative-ELBO convergence "
                             "tolerance; CmdStan default 0.01)")
    parser.add_argument("--vi-eval-elbo", type=int, default=None,
                        help="CmdStan eval_elbo (iterations between ELBO "
                             "evaluations; CmdStan default 100)")
    parser.add_argument("--vi-eta", type=float, default=None,
                        help="CmdStan eta (step-size scaling; only meaningful "
                             "with --vi-no-adapt)")
    parser.add_argument("--vi-adapt-iter", type=int, default=None,
                        help="CmdStan adapt_iter (eta-adaptation iterations; "
                             "CmdStan default 50)")
    parser.add_argument("--vi-no-adapt", action="store_true",
                        help="disable CmdStan eta adaptation (adapt_engaged=0)")
    parser.add_argument("--vi-show-console", action="store_true",
                        help="stream CmdStan's ADVI ELBO trace to stdout")
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--require-converged", action="store_true")
    parser.add_argument("--matched-init", action="store_true")
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

    # Matched-init control experiment: deterministic uniform-simplex init on the
    # constrained scale (Stan inits are constrained), identical to the Impulse
    # arm's NUTS start. Only replaces the init when the model does not already
    # supply one; the kernel is otherwise unchanged.
    matched_fn = getattr(impl, "matched_init_params", None)
    matched_init = None
    if args.matched_init and matched_fn is not None:
        matched_init = matched_fn(data)
        preview = ", ".join(
            f"{k}={np.asarray(v).reshape(-1)[:3]}" for k, v in matched_init.items()
        )
        print(f"Stan: --matched-init constrained point: {preview}", flush=True)

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

        # SVI (ADVI) path is separate from the NUTS path below (unchanged).
        if args.method in ("svi", "vi"):
            vi_main(args, stan_model, stan_data, impl, compile_time, module_path,
                    matched_init)
            return

        num_chains = args.num_chains
        if num_chains > 1:
            print(
                f"Stan: chains={num_chains} (native parallel_chains), "
                f"chain_method={args.chain_method}",
                flush=True,
            )

        # First trial. Matched init (when set) wins over any driver-supplied
        # per-trial init_params so every trial and every system starts from the
        # same deterministic point.
        first_init = (
            matched_init if matched_init is not None
            else data_dict.get("init_params")
        )
        t0 = time.time()
        samples = run_trial(
            stan_model, stan_data,
            args.num_warmup, args.num_samples, args.seed,
            args.step_size, args.max_tree_depth,
            args.adapt_step_size, args.adapt_mass_matrix,
            first_init, num_chains,
        )
        run_time = time.time() - t0

        write_output(args.output, samples, impl, compile_time, run_time,
                     num_chains, args.num_samples)
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
                # Matched init wins over any per-trial driver init.
                new_init = (
                    matched_init if matched_init is not None
                    else req.get("init_params")
                )

                t0 = time.time()
                new_samples = run_trial(
                    stan_model, stan_data,
                    args.num_warmup, args.num_samples, new_seed,
                    args.step_size, args.max_tree_depth,
                    args.adapt_step_size, args.adapt_mass_matrix,
                    new_init, num_chains,
                )
                new_run_time = time.time() - t0

                write_output(new_output_path, new_samples, impl, 0.0, new_run_time,
                             num_chains, args.num_samples)
                print(f"###DONE### {new_run_time*1000:.2f} ms", flush=True)

    finally:
        if os.path.exists(stan_file):
            os.unlink(stan_file)


if __name__ == "__main__":
    main()
