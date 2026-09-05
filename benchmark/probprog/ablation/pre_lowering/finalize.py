#!/usr/bin/env python3
"""Build the shipped raw-timing JSON from all batch dirs + print the summary."""
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as agg

DIR = "/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog/ablation/pre_lowering"
BATCHES = [("batch1", "logs_batch1", "2026-08-24T20:59:02-05:00"),
           ("batch2", "logs_batch2", "2026-08-24T21:17:39-05:00"),
           ("batch3", "logs_batch3", None)]

out = {"description": ("tab:prelowering controlled measurement, 2026-08-24. "
                       "Uniform two-arm protocol: arm A (No Pre-Opt) = flat expand path "
                       "(DEBUG_PROBPROG_DISABLE_OPT=true) + disable_licm_optimization_passes=true; "
                       "arm B (Pre-Opt) = flat expand path + full standard passes. "
                       "SICM off in both arms. 20 NUTS samples, adaptation off, CPU, "
                       "median of 5 runs per cell after 1 warm-up call, no thread caps."),
       "headline_batch": "batch2",
       "libReactantExtra_md5": "21a6bd08b37b54d2b8a20d8924a2bbb2",
       "libReactantExtra_mtime": "2026-08-24 14:59:21 -0500",
       "libReactantExtra_path": "oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so",
       "machine": "AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, 1 NUMA node",
       "julia": "1.11.7",
       "batches": {}}

present = []
for name, sub, _ in BATCHES:
    p = os.path.join(DIR, sub)
    if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "batch_meta.txt")):
        continue
    present.append(name)
    r = agg.run(p, None, "%s (%s)" % (name, sub))
    out["batches"][name] = r

# cross-batch summary
summary = []
for i, row in enumerate(out["batches"][present[0]]["rows"]):
    e = {"model": row["model"], "size": row["size"], "per_batch": {}}
    sps, bs, as_ = [], [], []
    for name in present:
        rr = out["batches"][name]["rows"][i]
        e["per_batch"][name] = {
            "pre_opt_median_ms": rr["pre_opt_B"]["median_ms"],
            "no_pre_opt_median_ms": rr["no_pre_opt_A"]["median_ms"],
            "speedup": rr["speedup"],
            "pre_opt_runs_ms": rr["pre_opt_B"]["times_ms"],
            "no_pre_opt_runs_ms": rr["no_pre_opt_A"]["times_ms"]}
        sps.append(rr["speedup"]); bs.append(rr["pre_opt_B"]["median_ms"])
        as_.append(rr["no_pre_opt_A"]["median_ms"])
    e["speedup_across_batches"] = {"min": min(sps), "median": statistics.median(sps),
                                   "max": max(sps)}
    e["pre_opt_ms_across_batches"] = {"min": min(bs), "median": statistics.median(bs), "max": max(bs)}
    e["no_pre_opt_ms_across_batches"] = {"min": min(as_), "median": statistics.median(as_), "max": max(as_)}
    summary.append(e)
out["cross_batch_summary"] = summary

out["ir_evidence"] = {
  "method": ("ablation/pre_lowering/analyze_ir.py classifies ops by stablehlo.while "
             "nesting depth. depth 0 = top level (InitHMC gradient), depth 1 = NUTS "
             "sampling while (and the InitHMC gradient's own model time loop), "
             "depth 3 = per leapfrog step outside the model time loop, "
             "depth 4 = innermost model time loop."),
  "Cubic Kernel (both sizes)": {
      "metric": "O(D^3) stablehlo.dot_general sites (FLOP-classified) by while-depth",
      "no_pre_opt_A": {"depth0": 1, "depth1": 3, "depth3": 1, "depth4": 3},
      "pre_opt_B":    {"depth0": 3, "depth3": 3},
      "reading": ("arm A keeps 3 O(D^3) matmuls in the innermost time loop (forward K "
                  "construction + 2 reverse-pass instances) and 3 more in the InitHMC "
                  "gradient's time loop; arm B has zero at either time-loop depth, with "
                  "the same 3+3 hoisted one while-level out. The time loop retains only "
                  "the O(D^2) 1xD @ DxD state updates in both arms.")},
  "Projected Dynamics": {
      "metric": "D x D elementwise/shape ops (the K = theta*A + I construction) by while-depth",
      "no_pre_opt_A": {"depth1": ["multiply", "add", "broadcast_in_dim"],
                       "depth4": ["multiply", "add", "broadcast_in_dim"]},
      "pre_opt_B":    {"depth1": ["add"], "depth4": ["add"],
                       "depth0": ["multiply", "add", "broadcast_in_dim"],
                       "depth3": ["multiply", "add", "broadcast_in_dim"]},
      "reading": ("no O(D^3) dot_general exists in either arm (the invariant is an O(D^2) "
                  "elementwise construction, which is why this row's speedup is the "
                  "smallest). Arm A rebuilds K inside the time loop every step; arm B "
                  "keeps only the loop-carried Kbar accumulate (one DxD add) there.")},
  "Heat Equation (D=100)": {
      "metric": "LAPACK custom_calls by result signature and while-depth",
      "no_pre_opt_A": {"potrf": {"depth0": 1, "depth1": 10, "depth3": 1, "depth4": 10},
                       "trsm": {"depth1": 60, "depth4": 60}},
      "pre_opt_B":    {"potrf": {"depth0": 1, "depth3": 1},
                       "trsm": {"depth1": 60, "depth4": 60}},
      "reading": ("22 potrf total in arm A, 20 of them inside the time loops; 2 total in "
                  "arm B, 0 in the time loops. dtrsm stays in the time loops in both "
                  "arms, as it must. Reproduces heat_inverse_RESULTS_2026-08-21.md "
                  "exactly on the current binary.")},
  "syevd": "0 in every measured cell of every batch (SICM-off proof)",
}

json.dump(out, open(os.path.join(DIR, "cubic_projected_RAW_2026-08-24.json"), "w"), indent=2)

print("\n### cross-batch (%s)" % ", ".join(present))
print("%-24s %-14s %26s %26s %20s" % ("Model", "Size", "Pre-Opt ms (min/med/max)",
                                      "NoPreOpt ms (min/med/max)", "Speedup min/med/max"))
for e in summary:
    b, a, s = e["pre_opt_ms_across_batches"], e["no_pre_opt_ms_across_batches"], e["speedup_across_batches"]
    print("%-24s %-14s %8.1f/%8.1f/%8.1f %8.1f/%8.1f/%8.1f %5.1f/%5.1f/%5.1f" %
          (e["model"], e["size"], b["min"], b["median"], b["max"],
           a["min"], a["median"], a["max"], s["min"], s["median"], s["max"]))
