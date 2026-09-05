#!/usr/bin/env python3
"""Validation gates + statistics extraction for one sicm_ablation run dir.

Reports, per arm:
  - all trial times, the last-trial statistic (what the paper figure uses) and
    the trial median;
  - n_eff [min, med, max] and Rhat [min, med, max] over all variables, post-warmup;
  - per-chain unique-draw counts and standard deviations of a scalar variable
    (the 2026-08-22 stuck-chain detector: 1 unique draw per chain or Rhat > 1e3
    invalidates the cell);
  - PLL[-1] per chain.

And, across the SICM / No SICM arms:
  - max |SICM - No SICM| over every sample of every variable (same shared trial
    seeds => the two arms should agree to the ULP floor);
  - max |PLL difference|.

Usage: postcheck.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys

import numpy as np
import xarray as xr


def summarize(run_dir):
    out = {"run_dir": run_dir}
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    out["config"] = cfg
    ds = xr.open_dataset(os.path.join(run_dir, "samples.nc"))
    metrics = xr.open_dataset(os.path.join(run_dir, "metrics.nc"))
    diag = xr.open_dataset(os.path.join(run_dir, "diagnostics.nc"))
    num_warmup = cfg.get("num_warmup", 0)

    ppls = [str(p) for p in ds.ppl.values]
    out["ppls"] = ppls
    arms = {}
    for i, name in enumerate(ppls):
        a = {}
        timing = metrics["timing"].values[i]      # [chain, phase]
        run_times = timing[:, 1]
        run_times = run_times[~np.isnan(run_times)]
        a["all_trial_times_s"] = [float(t) for t in run_times]
        a["compile_time_s"] = float(timing[0, 0])
        a["last_trial_s"] = float(run_times[-1])
        a["trial_median_s"] = float(np.median(run_times))
        a["steady_mean_s"] = float(np.mean(run_times[1:])) if len(run_times) > 1 else float(run_times[-1])

        sub = ds.isel(ppl=i).isel(draw=slice(num_warmup, None))
        neff = xr.apply_ufunc(lambda x: x, sub)  # keep type
        ess = __import__("arviz").ess(sub)
        rhat = __import__("arviz").rhat(sub)
        ev = np.concatenate([np.atleast_1d(ess[v].values).ravel() for v in ess.data_vars])
        rv = np.concatenate([np.atleast_1d(rhat[v].values).ravel() for v in rhat.data_vars])
        a["neff"] = [float(np.nanmin(ev)), float(np.nanmedian(ev)), float(np.nanmax(ev))]
        a["rhat"] = [float(np.nanmin(rv)), float(np.nanmedian(rv)), float(np.nanmax(rv))]
        a["neff_per_s_last_trial"] = [float(x) / a["last_trial_s"] for x in a["neff"]]

        # movement: per-chain unique draws and std of every scalar variable
        mv = {}
        for v in sub.data_vars:
            arr = sub[v].values
            arr2 = arr.reshape(arr.shape[0], arr.shape[1], -1)
            uniq = [int(len(np.unique(np.round(arr2[c], 15)))) for c in range(arr2.shape[0])]
            stds = [float(np.nanmax(np.nanstd(arr2[c], axis=0))) for c in range(arr2.shape[0])]
            mv[str(v)] = {"unique_per_chain": uniq, "max_std_per_chain": stds,
                          "n_draws": int(arr2.shape[1])}
        a["movement"] = mv

        pll = metrics["pll"].values[i]  # [chain, draw]
        a["pll_last_per_chain"] = [
            float(p[~np.isnan(p)][-1]) if np.any(~np.isnan(p)) else float("nan")
            for p in pll
        ]
        arms[name] = a
    out["arms"] = arms

    # cross-arm agreement
    if len(ppls) >= 2:
        i0, i1 = 0, 1
        agree = {}
        for v in ds.data_vars:
            x0 = ds[v].isel(ppl=i0).values
            x1 = ds[v].isel(ppl=i1).values
            d = np.nanmax(np.abs(x0 - x1))
            agree[str(v)] = float(d)
        p0 = metrics["pll"].values[i0]
        p1 = metrics["pll"].values[i1]
        agree["__pll__"] = float(np.nanmax(np.abs(p0 - p1)))
        out["arm_agreement_max_abs_diff"] = agree
    return out


def main():
    res = []
    for run_dir in sys.argv[1:]:
        s = summarize(run_dir)
        res.append(s)
        print("=" * 78)
        print("RUN", run_dir)
        print("  model:", s["config"]["model"]["class"], s["config"]["model"].get("args"))
        print("  iterations=%s num_warmup=%s trials=%s"
              % (s["config"]["iterations"], s["config"].get("num_warmup"),
                 s["config"]["trials"]))
        for name, a in s["arms"].items():
            print("  -- arm %-10s compile=%.2fs  trials=%s  LAST=%.3fs  MEDIAN=%.3fs"
                  % (name, a["compile_time_s"],
                     ";".join("%.3f" % t for t in a["all_trial_times_s"]),
                     a["last_trial_s"], a["trial_median_s"]))
            print("     n_eff[min,med,max] = [%.1f, %.1f, %.1f]   Rhat[min,med,max] = [%.4f, %.4f, %.4f]"
                  % tuple(a["neff"] + a["rhat"]))
            print("     PLL[-1] per chain:", ["%.4f" % x for x in a["pll_last_per_chain"]])
            for v, m in a["movement"].items():
                print("     move %-12s uniq/chain=%s (of %d draws)  max_std/chain=%s"
                      % (v, m["unique_per_chain"], m["n_draws"],
                         ["%.3g" % x for x in m["max_std_per_chain"]]))
        if "arm_agreement_max_abs_diff" in s:
            print("  -- arm agreement (max |arm0 - arm1|):")
            for v, d in s["arm_agreement_max_abs_diff"].items():
                print("     %-14s %.6g" % (v, d))
        # gates
        bad = []
        for name, a in s["arms"].items():
            if a["rhat"][2] > 1e3:
                bad.append("%s: Rhat max %.3g > 1e3" % (name, a["rhat"][2]))
            for v, m in a["movement"].items():
                if min(m["unique_per_chain"]) <= 1 and m["n_draws"] > 1:
                    bad.append("%s: %s has a frozen chain (uniq=%s)"
                               % (name, v, m["unique_per_chain"]))
        print("  GATE:", "FAIL -> " + "; ".join(bad) if bad else "pass (chains move, Rhat sane)")
    outpath = os.environ.get("POSTCHECK_JSON")
    if outpath:
        with open(outpath, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", outpath)


if __name__ == "__main__":
    main()
