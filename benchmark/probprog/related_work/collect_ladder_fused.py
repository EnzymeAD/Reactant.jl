#!/usr/bin/env python3
"""Collect the FUSED (benchmark B) heat-inverse + GP-prior ladder run.

Sibling of collect_ladder.py, which collects benchmark A (GP Poisson regression).
Separate file so benchmark A stays byte-reproducible.

Usage: python collect_ladder_fused.py <run dir> <stan diag dir> <out json>
                                      [--impulse-run-dir DIR]

Benchmark B has SIX Stan rungs and FOUR Impulse arms. Arm R (region pipeline with
SICM at zero iterations) is present because R != B: the region path alone sheds
per-step work, so quoting tier 2 as C-vs-B would credit SICM with gains the region
path produced. Tier 2 is C vs R. See
oopsla/evidence/related_work/fused_ladder_pilot_2026-08-25.md ADDENDUM B.
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
if impulse_run_dir:
    m_imp = xr.open_dataset(os.path.join(impulse_run_dir, "metrics.nc"))
    s_imp = xr.open_dataset(os.path.join(impulse_run_dir, "samples.nc"))
else:
    m_imp, s_imp = m, s
try:
    cfg = json.loads(open(os.path.join(run_dir, "config.json")).read(), strict=False)
except (json.JSONDecodeError, FileNotFoundError):
    cfg = {}

variant_of = {
    "Fused Stan (naive)": "naive",
    "Fused Stan (SlicStan)": "slicstan",
    "Fused Stan (SlicStan, substitution)": "slicstan_subst",
    "Fused Stan (expert A, LICM)": "expert_a",
    "Fused Stan (expert B, +CholScale)": "expert_b",
    "Fused Stan (expert C, +EigenLift)": "expert_c",
}

ORDER = [  # (legend name, short label, system, what is off the per-gradient path, tier)
    ("Fused Stan (naive)", "Stan, naive", "Stan",
     "nothing hoisted", "-"),
    ("Fused Stan (SlicStan)", "Stan, SlicStan-translated", "Stan",
     "kernel build -> transformed data", "tier 0"),
    ("Fused Stan (SlicStan, substitution)", "Stan, SlicStan (own solve spelling)", "Stan",
     "same placement, hand substitution instead of mdivide", "tier 0"),
    ("Fused Stan (expert A, LICM)", "Stan, expert A", "Stan",
     "chol(M(kappa)) hoisted out of the time loop", "tier 1"),
    ("Fused Stan (expert B, +CholScale)", "Stan, expert B", "Stan",
     "+ chol(K) -> transformed data (CholScale by hand)", "tier 2a"),
    ("Fused Stan (expert C, +EigenLift)", "Stan, expert C", "Stan",
     "+ eig(Lap) -> transformed data (EigenLift by hand)", "tier 2b"),
    ("Impulse (arm A, no pre-opt)", "Impulse, LICM patterns off", "Impulse",
     "chol(M) still inside the time loop (SlicStan-equivalent)", "-"),
    ("Impulse (arm B, std passes)", "Impulse, standard passes", "Impulse",
     "chol(M) hoisted by LICM; both chols per gradient", "tier 1"),
    ("Impulse (arm R, region only)", "Impulse, region pipeline, SICM off", "Impulse",
     "region restructuring only; SICM at zero iterations", "baseline for tier 2"),
    ("Impulse (arm C, SICM)", "Impulse, SICM", "Impulse",
     "no factorization in the gradient path (CholScale + EigenLift)", "tier 2"),
]


def diag_us_per_grad(variant):
    vals = []
    for f in glob.glob(os.path.join(diag_dir, f"{variant}.heat_gp_fused_seed*.json")):
        j = json.load(open(f))
        if j.get("elapsed_sampling_s") and j.get("n_leapfrog_sampling_total"):
            vals.append(1e6 * j["elapsed_sampling_s"] / j["n_leapfrog_sampling_total"])
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


rungs = []
for name, label, system, hoisted, tier in ORDER:
    src = m_imp if system == "Impulse" else m
    if name not in src.ppl.values:
        continue
    t = src.timing.sel(ppl=name).values
    infer = t[:, 1]; compile_t = float(t[0, 0])
    neff = float(src.overall_neff_per_time.sel(ppl=name, percentile="median"))
    pll = float(np.nanmean(src.pll.sel(ppl=name).isel(draw=-1).values))
    us, n = (diag_us_per_grad(variant_of[name]) if system == "Stan" else (None, 0))
    rungs.append({"name": name, "label": label, "system": system, "hoisted": hoisted,
                  "tier": tier, "compile_s": compile_t,
                  "wall_s_per_trial": [float(x) for x in infer],
                  "wall_s_median": float(np.median(infer)), "neff_per_s": neff,
                  "pll_final": pll, "us_per_grad": us, "us_per_grad_n_trials": n})


def maxdiff(a, b, ds):
    out = 0.0
    for c in ds.chain.values:
        for v in ("z_tau", "z_kappa", "u_tilde"):
            if v not in ds:
                continue
            out = max(out, float(np.max(np.abs(
                ds[v].sel(ppl=a).isel(chain=int(c)).values
                - ds[v].sel(ppl=b).isel(chain=int(c)).values))))
    return out


identity = {}
PAIRS = [("Fused Stan (naive)", "Fused Stan (SlicStan)", s),
         ("Fused Stan (naive)", "Fused Stan (SlicStan, substitution)", s),
         ("Fused Stan (naive)", "Fused Stan (expert A, LICM)", s),
         ("Fused Stan (naive)", "Fused Stan (expert B, +CholScale)", s),
         ("Fused Stan (naive)", "Fused Stan (expert C, +EigenLift)", s),
         ("Impulse (arm A, no pre-opt)", "Impulse (arm B, std passes)", s_imp),
         ("Impulse (arm B, std passes)", "Impulse (arm R, region only)", s_imp)]
for a, b, ds in PAIRS:
    if a in ds.ppl.values and b in ds.ppl.values:
        identity[f"{a} vs {b}"] = maxdiff(a, b, ds)

by = {r["name"]: r for r in rungs}
base = by.get("Fused Stan (naive)")
for r in rungs:
    r["speedup_vs_stan_naive_wall"] = (base["wall_s_median"] / r["wall_s_median"]) if base else None
    r["speedup_vs_stan_naive_neff"] = (r["neff_per_s"] / base["neff_per_s"]) if base else None


def ratio(a, b, key="wall_s_median"):
    if a in by and b in by and by[b][key]:
        return by[a][key] / by[b][key]
    return None


gaps = {
    "stan_tier0_naive_over_slicstan": ratio("Fused Stan (naive)", "Fused Stan (SlicStan)"),
    "stan_tier1_slicstan_over_expertA": ratio("Fused Stan (SlicStan)", "Fused Stan (expert A, LICM)"),
    "stan_cholscale_expertA_over_expertB": ratio("Fused Stan (expert A, LICM)", "Fused Stan (expert B, +CholScale)"),
    "stan_eigenlift_expertB_over_expertC": ratio("Fused Stan (expert B, +CholScale)", "Fused Stan (expert C, +EigenLift)"),
    "stan_tier2_expertA_over_expertC": ratio("Fused Stan (expert A, LICM)", "Fused Stan (expert C, +EigenLift)"),
    "impulse_tier1_armA_over_armB": ratio("Impulse (arm A, no pre-opt)", "Impulse (arm B, std passes)"),
    "impulse_region_armB_over_armR": ratio("Impulse (arm B, std passes)", "Impulse (arm R, region only)"),
    "impulse_tier2_armR_over_armC": ratio("Impulse (arm R, region only)", "Impulse (arm C, SICM)"),
    "impulse_armB_over_armC_DO_NOT_QUOTE_AS_TIER2": ratio("Impulse (arm B, std passes)", "Impulse (arm C, SICM)"),
}

out = {"_provenance": {
           "benchmark": "B (fused heat-inverse + GP prior)",
           "run_dir": run_dir, "impulse_run_dir": impulse_run_dir, "diag_dir": diag_dir,
           "model": cfg.get("model"), "iterations": cfg.get("iterations"),
           "num_warmup": cfg.get("num_warmup"), "trials": cfg.get("trials"),
           "kernel": "Matern-1/2, exp(-|s_i - s_j|/rho); SE is not positive definite on this grid",
           "tier2_attribution": "quote Impulse tier 2 as arm C vs arm R, NOT vs arm B (R != B)",
           "metric": "overall n_eff/s median over trials; wall = infer phase per trial; "
                     "us/grad = CmdStan sampling elapsed / total n_leapfrog__ (Stan rungs only)",
           "chain_identity_max_abs_diff": identity},
       "gaps": gaps, "rungs": rungs}
json.dump(out, open(out_path, "w"), indent=1)

print(f"{'arm':38s} {'compile':>8s} {'wall/trial (s)':>20s} {'neff/s':>8s} {'us/grad':>9s} {'PLL':>10s} {'x vs naive':>11s}")
for r in rungs:
    w = "/".join(f"{x:.1f}" for x in r["wall_s_per_trial"])
    us = f"{r['us_per_grad']:.0f}" if r["us_per_grad"] else "-"
    sp = f"{r['speedup_vs_stan_naive_wall']:.2f}" if r["speedup_vs_stan_naive_wall"] else "-"
    print(f"{r['name']:38s} {r['compile_s']:8.1f} {w:>20s} {r['neff_per_s']:8.2f} {us:>9s} {r['pll_final']:10.2f} {sp:>11s}")
print("\ngaps:")
for k, v in gaps.items():
    print(f"  {k:52s} {v if v is None else round(v, 2)}")
print("\nchain identity (max |diff|):", identity)
print("wrote", out_path)
