#!/usr/bin/env python3
"""Collect the SlicStan optimization-ladder run into the paper data JSON.

Usage: python collect_ladder.py <pplbench run dir> <stan diag dir> <out json> [--print]
                                [--impulse-run-dir DIR]

``--impulse-run-dir`` pins the Impulse rungs to a DIFFERENT pplbench run than the
Stan rungs. Needed for panel A of the two-panel figure: the Stan rungs stand from
outputs/2026-08-23_00:24:17, but the Impulse rungs must come from the pinned
single-thread re-run outputs/2026-08-24_04:06:42 (471.3 / 472.1 / 7.4 s), because
the 08-23 Impulse rungs were taken with threads uncapped. Omit the flag and every
rung is read from the single positional run dir, exactly as before.
Reads metrics.nc (timing, overall_neff_per_time, pll) and samples.nc (chain
identity checks) written by the pplbench driver, plus the per-trial Stan NUTS
diagnostics side files (LADDER_DIAG_DIR) for a leapfrog-normalized
per-gradient time on the Stan rungs. Writes paper/figures/slicstan_ladder_data.json
-style output (with a _provenance block) and prints a table.
"""
import glob, json, os, sys
import numpy as np
import xarray as xr

argv = list(sys.argv[1:])
impulse_run_dir = None
if "--impulse-run-dir" in argv:
    i = argv.index("--impulse-run-dir")
    impulse_run_dir = argv[i + 1]
    del argv[i:i + 2]
run_dir, diag_dir, out_path = argv[0:3]
m = xr.open_dataset(os.path.join(run_dir, "metrics.nc"))
s = xr.open_dataset(os.path.join(run_dir, "samples.nc"))
# Optional second source for the Impulse rungs only.
if impulse_run_dir:
    m_imp = xr.open_dataset(os.path.join(impulse_run_dir, "metrics.nc"))
    s_imp = xr.open_dataset(os.path.join(impulse_run_dir, "samples.nc"))
else:
    m_imp, s_imp = m, s
try:
    cfg = json.loads(open(os.path.join(run_dir, "config.json")).read(), strict=False)
except json.JSONDecodeError:
    cfg = json.load(open(sys.argv[5]))  # fallback: source config path as 5th arg
variant_of = {"Stan (naive)": "naive", "Stan (SlicStan)": "slicstan",
              "Stan (SlicStan, locals)": "slicstan_local", "Stan (expert)": "expert"}

ORDER = [  # (legend name, short label, system, what is hoisted off the per-gradient path)
    ("Stan (naive)", "Stan, naive", "Stan", "nothing hoisted"),
    ("Stan (SlicStan)", "Stan, SlicStan-translated", "Stan", "kernel build → transformed data"),
    ("Stan (SlicStan, locals)", "Stan, SlicStan (model-local)", "Stan", "same placement, no CSV output of L"),
    ("Stan (expert)", "Stan, expert rewrite", "Stan", "chol(K) → transformed data (CholScale by hand)"),
    ("Impulse (no pre-opt)", "Impulse, LICM patterns off", "Impulse", "kernel build hoisted anyway (CSE vs. init eval)"),
    ("Impulse (std passes)", "Impulse, standard passes", "Impulse", "kernel build hoisted; chol per gradient"),
    ("Impulse (SICM)", "Impulse, SICM", "Impulse", "chol(K) hoisted (CholScale rewrite)"),
]

def diag_us_per_grad(variant):
    vals = []
    for f in glob.glob(os.path.join(diag_dir, f"{variant}.gp_pois_regr_scaled_seed*.json")):
        j = json.load(open(f))
        if j.get("elapsed_sampling_s") and j.get("n_leapfrog_sampling_total"):
            vals.append(1e6 * j["elapsed_sampling_s"] / j["n_leapfrog_sampling_total"])
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)

rungs = []
for name, label, system, hoisted in ORDER:
    src = m_imp if system == "Impulse" else m
    if name not in src.ppl.values:
        continue
    t = src.timing.sel(ppl=name).values  # (chain/trial, phase)
    infer = t[:, 1]; compile_t = float(t[0, 0])
    neff = float(src.overall_neff_per_time.sel(ppl=name, percentile="median"))
    pll = float(np.nanmean(src.pll.sel(ppl=name).isel(draw=-1).values))
    us, n = (diag_us_per_grad(variant_of[name]) if system == "Stan" else (None, 0))
    rungs.append({"name": name, "label": label, "system": system, "hoisted": hoisted,
                  "compile_s": compile_t, "wall_s_per_trial": [float(x) for x in infer],
                  "wall_s_median": float(np.median(infer)), "neff_per_s": neff, "pll_final": pll,
                  "us_per_grad": us, "us_per_grad_n_trials": n})

def maxdiff(a, b, ds):
    out = 0.0
    for c in ds.chain.values:
        for v in ("alpha", "f_tilde"):
            out = max(out, float(np.max(np.abs(ds[v].sel(ppl=a).isel(chain=int(c)).values - ds[v].sel(ppl=b).isel(chain=int(c)).values))))
    return out
identity = {}
for a, b, ds in [("Stan (naive)", "Stan (SlicStan)", s),
                 ("Stan (naive)", "Stan (SlicStan, locals)", s),
                 ("Impulse (no pre-opt)", "Impulse (std passes)", s_imp)]:
    if a in ds.ppl.values and b in ds.ppl.values:
        identity[f"{a} vs {b}"] = maxdiff(a, b, ds)

base = next((r for r in rungs if r["name"] == "Stan (naive)"), None)
for r in rungs:
    r["speedup_vs_stan_naive_wall"] = (base["wall_s_median"] / r["wall_s_median"]) if base else None
    r["speedup_vs_stan_naive_neff"] = (r["neff_per_s"] / base["neff_per_s"]) if base else None

out = {"_provenance": {"run_dir": run_dir, "impulse_run_dir": impulse_run_dir, "diag_dir": diag_dir, "model": cfg.get("model"),
                       "iterations": cfg.get("iterations"), "num_warmup": cfg.get("num_warmup"), "trials": cfg.get("trials"),
                       "metric": "overall n_eff/s median over trials (pplbench driver, arviz ess); wall = infer phase per trial; us/grad = CmdStan sampling-phase elapsed / total n_leapfrog__ (Stan rungs only)",
                       "chain_identity_max_abs_diff": identity},
       "rungs": rungs}
json.dump(out, open(out_path, "w"), indent=1)
print(f"{'arm':30s} {'compile':>8s} {'wall/trial (s)':>20s} {'neff/s':>9s} {'us/grad':>8s} {'PLL':>8s} {'x vs naive(wall)':>16s}")
for r in rungs:
    w = "/".join(f"{x:.1f}" for x in r["wall_s_per_trial"])
    us = f"{r['us_per_grad']:.0f}" if r["us_per_grad"] else "-"
    sp = f"{r['speedup_vs_stan_naive_wall']:.2f}" if r["speedup_vs_stan_naive_wall"] else "-"
    print(f"{r['name']:30s} {r['compile_s']:8.1f} {w:>20s} {r['neff_per_s']:9.1f} {us:>8s} {r['pll_final']:8.2f} {sp:>16s}")
print("chain identity (max |diff|):", identity)
print("wrote", out_path)
