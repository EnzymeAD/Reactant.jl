#!/usr/bin/env python3
"""Aggregate a v2 pre-lowering batch (the four figure rows) into a raw-timing
JSON plus a verification report (cross-arm accept identity, sample agreement,
syevd). Companion to aggregate.py, which covers the superseded Cubic Kernel /
Projected Dynamics row set."""
import json
import math
import os
import re
import statistics
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(DIR, "..", "..", "..", "..", "deps", "ReactantExtra",
                  "bazel-bin", "libReactantExtra.so")


def _so_md5():
    """md5 of the binary these numbers were measured on, read at aggregation
    time from the file itself."""
    import hashlib
    try:
        h = hashlib.md5()
        with open(SO, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

# (row label, size label, arm-B log tag, arm-A log tag, heat parity scale or None,
#  log dir override or None meaning "the batch dir passed on the command line")
#
# The Heat rows come from the quiet 2026-08-24 batch 2, not from the 2026-08-25
# batch: see advdiff_star_RESULTS_2026-08-25.md ("single-build, two-batch").
# Under the frozen-adapted protocol (2026-08-25) ALL four rows are measured in
# the same batch, so no per-row directory override is needed.
ROWS = [
    ("Heat Equation",         "D=100, T=100", "heat_s1_B",  "heat_s1_A",  None, None),
    ("Heat Equation (large)", "D=300, T=100", "heat_s3_B",  "heat_s3_A",  None, None),
    ("Advection-Diffusion",   "D=200, T=100", "advdiff_B",  "advdiff_A",  None, None),
    ("STAR (ILINet)",         "N=47, T=104",  "star_B",     "star_A",     None, None),
    # Added 2026-09-08: the two synthetic rows retired from the figure in
    # 2026-08-25, re-measured under the frozen-adapted protocol for the first
    # time (their harnesses only had the naive fixed-step path until now).
    ("Cubic Kernel",          "D=60, T=50",   "cubic_s1_B", "cubic_s1_A", None, None),
    ("Cubic Kernel (large)",  "D=100, T=100", "cubic_s2_B", "cubic_s2_A", None, None),
    ("Projected Dynamics",    "D=100, T=100", "projected_B", "projected_A", None, None),
]
WARMUP_TAG = {"Heat Equation": "heat_s1_warmup",
              "Heat Equation (large)": "heat_s3_warmup",
              "Advection-Diffusion": "advdiff_warmup",
              "STAR (ILINet)": "star_warmup",
              "Cubic Kernel": "cubic_s1_warmup",
              "Cubic Kernel (large)": "cubic_s2_warmup",
              "Projected Dynamics": "projected_warmup"}
CELL_JSON = {
    "advdiff_B": "advdiff_scale1_armB.json",
    "advdiff_A": "advdiff_scale1_armA.json",
    "star_B": "star_recur_armB.json",
    "star_A": "star_recur_armA.json",
    "heat_s1_B": "heat_scale1_armB.json", "heat_s1_A": "heat_scale1_armA.json",
    "heat_s3_B": "heat_scale3_armB.json", "heat_s3_A": "heat_scale3_armA.json",
    "cubic_s1_B": "cubic_small_armB.json", "cubic_s1_A": "cubic_small_armA.json",
    "cubic_s2_B": "cubic_large_armB.json", "cubic_s2_A": "cubic_large_armA.json",
    # note the `armlarge` infix: it is in the harness's cell-file name only
    "projected_B": "projected_armlarge_armB.json",
    "projected_A": "projected_armlarge_armA.json",
}
STATE_JSON = {"Heat Equation": "heat_scale1.json",
              "Heat Equation (large)": "heat_scale3.json",
              "Advection-Diffusion": "advdiff_scale1.json",
              "STAR (ILINet)": "star_recur.json",
              "Cubic Kernel": "cubic_small.json",
              "Cubic Kernel (large)": "cubic_large.json",
              "Projected Dynamics": "projected.json"}

# Problem-size sweep, 2026-09-10. Selected with AGG_ROWS=sweep so the default
# behaviour of this script (the frozen figure-9 rows) is untouched.
SWEEP_ROWS = [
    ("Heat Equation (large)",  "D=300, T=100",  "heat_s3_B",    "heat_s3_A",    None, None),
    ("Heat Equation T=400",    "D=300, T=400",  "heat_s7_B",    "heat_s7_A",    None, None),
    ("Heat Equation D=1000",   "D=1000, T=100", "heat_s6_B",    "heat_s6_A",    None, None),
    ("Advection-Diffusion",    "D=200, T=100",  "advdiff_B",    "advdiff_A",    None, None),
    ("Adv-Diff D=500",         "D=500, T=250",  "advdiff_s3_B", "advdiff_s3_A", None, None),
    ("Adv-Diff D=1000",        "D=1000, T=500", "advdiff_s4_B", "advdiff_s4_A", None, None),
]
SWEEP_CELL_JSON = {
    "heat_s6_B": "heat_scale6_armB.json", "heat_s6_A": "heat_scale6_armA.json",
    "heat_s7_B": "heat_scale7_armB.json", "heat_s7_A": "heat_scale7_armA.json",
    "advdiff_s3_B": "advdiff_scale3_armB.json",
    "advdiff_s3_A": "advdiff_scale3_armA.json",
    "advdiff_s4_B": "advdiff_scale4_armB.json",
    "advdiff_s4_A": "advdiff_scale4_armA.json",
}
SWEEP_STATE_JSON = {"Heat Equation T=400": "heat_scale7.json",
                    "Heat Equation D=1000": "heat_scale6.json",
                    "Adv-Diff D=500": "advdiff_scale3.json",
                    "Adv-Diff D=1000": "advdiff_scale4.json"}
SWEEP_WARMUP_TAG = {"Heat Equation T=400": "heat_s7_warmup",
                    "Heat Equation D=1000": "heat_s6_warmup",
                    "Adv-Diff D=500": "advdiff_s3_warmup",
                    "Adv-Diff D=1000": "advdiff_s4_warmup"}
if os.environ.get("AGG_ROWS") == "sweep":
    ROWS = SWEEP_ROWS
    CELL_JSON.update(SWEEP_CELL_JSON)
    STATE_JSON.update(SWEEP_STATE_JSON)
    WARMUP_TAG.update(SWEEP_WARMUP_TAG)

ALL_RE = re.compile(r"All:\s*\[([^\]]*)\]ms")
MED_RE = re.compile(r"Median \(20 NUTS samples\):\s*([0-9.]+)ms")
GEO_RE = re.compile(r"Geomean \(20 NUTS samples\):\s*([0-9.]+)ms")
COMP_RE = re.compile(r"Compile:\s*([0-9.]+)s")
IR_RE = re.compile(r"^\s+(\S+)\s+in=(\d+)\s+out=(\d+)\s+total=(\d+)\s*$")


def parse_log(logs, tag):
    path = os.path.join(logs, tag + ".log")
    if not os.path.exists(path):
        # A partial batch (a row not yet measured) must not abort the whole
        # aggregation; the row is reported as failed instead.
        return {"log": tag + ".log", "times_ms": None, "median_ms": None,
                "geomean_ms": None, "compile_s": None, "ir": {}, "missing": True}
    txt = open(path, errors="replace").read()
    m = ALL_RE.search(txt)
    d = {"log": tag + ".log",
         "times_ms": [float(x) for x in m.group(1).split(",")] if m else None,
         "median_ms": float(MED_RE.search(txt).group(1)) if MED_RE.search(txt) else None,
         "geomean_ms": float(GEO_RE.search(txt).group(1)) if GEO_RE.search(txt) else None,
         "compile_s": float(COMP_RE.search(txt).group(1)) if COMP_RE.search(txt) else None,
         "ir": {}}
    for line in txt.splitlines():
        mm = IR_RE.match(line)
        if mm:
            d["ir"][mm.group(1)] = {"in": int(mm.group(2)), "out": int(mm.group(3)),
                                    "total": int(mm.group(4))}
    return d


def load_cell(logs, tag):
    d = parse_log(logs, tag)
    fn = CELL_JSON.get(tag)
    p = os.path.join(logs, "cells", fn) if fn else None
    if p and os.path.exists(p):
        j = json.load(open(p))
        for k in ("samples", "accepted", "diverging", "log_densities", "verify_seed"):
            d[k] = j.get(k)
        d["times_ms"] = j["times_ms"]
        d["median_ms"] = j["median_ms"]
        d["compile_s"] = j["compile_s"]
        t = j["times_ms"]
        d["geomean_ms"] = math.exp(sum(math.log(x) for x in t) / len(t))
        for k in ("stiffness_ratio", "advection_cfl", "N", "T", "D", "n_edges", "form"):
            if k in j:
                d[k] = j[k]
    return d


def run(logs, label):
    out = {"batch": label,
           "meta": open(os.path.join(logs, "batch_meta.txt"), errors="replace").read(),
           "rows": []}
    print("\n### %s" % label)
    print("%-22s %-13s %9s %9s %10s %10s %8s %8s %6s" %
          ("Model", "Size", "B med", "B geo", "A med", "A geo",
           "sp(med)", "sp(geo)", "dgeo%"))
    for label_, size, tb, ta, scale, dir_override in ROWS:
        d = dir_override or logs
        b, a = load_cell(d, tb), load_cell(d, ta)
        if not (a["median_ms"] and b["median_ms"]):
            print("%-24s %-14s  *** CELL FAILED (no median in log) ***" % (label_, size))
            out["rows"].append({"model": label_, "size": size, "failed": True,
                                "pre_opt_B": b, "no_pre_opt_A": a})
            continue
        sp = a["median_ms"] / b["median_ms"]
        ver = {"syevd_A": a["ir"].get("syevd", {}).get("total"),
               "syevd_B": b["ir"].get("syevd", {}).get("total")}
        if a.get("samples"):
            ver["accept_identical"] = (a["accepted"] == b["accepted"])
            ver["diverging_identical"] = (a["diverging"] == b["diverging"])
            ver["n_accepted"] = sum(a["accepted"])
            ver["n_diverging"] = sum(a["diverging"])
            ver["max_abs_sample_diff"] = max(
                abs(x - y) for ra, rb in zip(a["samples"], b["samples"])
                for x, y in zip(ra, rb))
            ver["max_abs_logdensity_diff"] = max(
                abs(x - y) for x, y in zip(a["log_densities"], b["log_densities"]))
            ver["verify_seed"] = a["verify_seed"]
        elif scale:
            pp = {}
            for arm in "AB":
                fp = os.path.join(d, "parity", "heat_s%d_arm%s.json" % (scale, arm))
                pp[arm] = json.load(open(fp)) if os.path.exists(fp) else None
            if pp["A"] and pp["B"]:
                ver["accept_identical"] = (pp["A"]["accepted"] == pp["B"]["accepted"])
                ver["max_abs_sample_diff"] = max(
                    abs(x - y) for ra, rb in zip(pp["A"]["samples"], pp["B"]["samples"])
                    for x, y in zip(ra, rb))
                ver["verify_seed"] = pp["A"]["seed"]
        sprb = (max(b["times_ms"]) - min(b["times_ms"])) / b["median_ms"]
        spra = (max(a["times_ms"]) - min(a["times_ms"])) / a["median_ms"]
        sp_geo = a["geomean_ms"] / b["geomean_ms"]
        # Flag when the median headline and the geomean of the same executions
        # disagree by more than a few percent.
        dgeo = 100 * abs(sp_geo - sp) / sp
        # Adapted-state provenance for this row.
        st_path = os.path.join(DIR, "adapted_states", STATE_JSON[label_])
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        wtag = WARMUP_TAG.get(label_)
        wlog = os.path.join(d, wtag + ".log") if wtag else None
        wcompile = None
        if wlog and os.path.exists(wlog):
            m = COMP_RE.search(open(wlog, errors="replace").read())
            wcompile = float(m.group(1)) if m else None
        out["rows"].append({"model": label_, "size": size,
                            "speedup": sp, "speedup_geomean": sp_geo,
                            "geomean_vs_median_pct": dgeo,
                            "pre_opt_B": b, "no_pre_opt_A": a, "verification": ver,
                            "spread_B_frac": sprb, "spread_A_frac": spra,
                            "adapted_state": {k: st.get(k) for k in
                                              ("adapted_step_size", "initial_step_size",
                                               "inverse_mass_rank", "num_warmup",
                                               "produced_by_arm", "warmup_seed",
                                               "post_warmup_position",
                                               "adapted_inverse_mass")},
                            "warmup_compile_s": wcompile})
        print("%-22s %-13s %9.1f %9.1f %10.1f %10.1f %7.2fx %7.2fx %5.1f%%  %s" %
              (label_, size, b["median_ms"], b["geomean_ms"],
               a["median_ms"], a["geomean_ms"], sp, sp_geo, dgeo,
               json.dumps({k: ver[k] for k in ("accept_identical",
                                               "max_abs_sample_diff",
                                               "syevd_A", "syevd_B") if k in ver})))
    return out


if __name__ == "__main__":
    batches, present = {}, []
    for name in sys.argv[1:]:
        p = os.path.join(DIR, name)
        if not os.path.exists(os.path.join(p, "batch_meta.txt")):
            print("skip (no batch_meta.txt): %s" % name)
            continue
        present.append(name)
        batches[name] = run(p, name)

    summary = []
    for i in range(len(ROWS)):
        e = {"model": ROWS[i][0], "size": ROWS[i][1], "per_batch": {}}
        sps, bs, as_ = [], [], []
        for name in present:
            rr = batches[name]["rows"][i]
            if rr.get("failed"):
                continue
            e["per_batch"][name] = {
                "pre_opt_median_ms": rr["pre_opt_B"]["median_ms"],
                "no_pre_opt_median_ms": rr["no_pre_opt_A"]["median_ms"],
                "speedup": rr["speedup"],
                "pre_opt_runs_ms": rr["pre_opt_B"]["times_ms"],
                "no_pre_opt_runs_ms": rr["no_pre_opt_A"]["times_ms"]}
            sps.append(rr["speedup"])
            bs.append(rr["pre_opt_B"]["median_ms"])
            as_.append(rr["no_pre_opt_A"]["median_ms"])
        if sps:
            e["speedup_across_batches"] = {"min": min(sps), "median": statistics.median(sps),
                                           "max": max(sps)}
            e["pre_opt_ms_across_batches"] = {"min": min(bs), "median": statistics.median(bs),
                                              "max": max(bs)}
            e["no_pre_opt_ms_across_batches"] = {"min": min(as_),
                                                 "median": statistics.median(as_),
                                                 "max": max(as_)}
        summary.append(e)

    # Compile-time geomean across the four benchmarks, per arm.
    comp = {"pre_opt_B": [], "no_pre_opt_A": []}
    for name in present:
        for rr in batches[name]["rows"]:
            if rr.get("failed"):
                continue
            for k in comp:
                if rr[k].get("compile_s"):
                    comp[k].append(rr[k]["compile_s"])
    compile_geo = {k: (math.exp(sum(math.log(x) for x in v) / len(v)) if v else None)
                   for k, v in comp.items()}
    print("\ncompile time geomean across the four benchmarks:")
    for k, v in compile_geo.items():
        print("  %-12s %s  (n=%d, values %s)" %
              (k, ("%.1f s" % v) if v else "n/a", len(comp[k]),
               [round(x, 1) for x in comp[k]]))

    out = {
        "compile_time_geomean_s": compile_geo,
        "compile_times_s": comp,
        "description": (
            "tab:prelowering figure rows, 2026-08-25. Uniform two-arm protocol: "
            "arm A (No Pre-Opt) = flat expand path (DEBUG_PROBPROG_DISABLE_OPT=true) "
            "+ disable_licm_optimization_passes=true; arm B (Pre-Opt) = flat expand "
            "path + full standard passes. SICM off in both arms. 20 NUTS samples, "
            "adaptation off, CPU, median of 5 runs per cell after 1 warm-up call, "
            "no thread caps. FROZEN-ADAPTED-STATE: each model was first warmed up "
            "with adaptation ON (step size + mass matrix) in the Pre-Opt arm, and "
            "BOTH arms are then measured with adaptation OFF from that same saved "
            "step size, inverse mass matrix and post-warmup position, with matched "
            "seeds, so the arms walk bit-identical trajectories."),
        "headline_batch": present[0] if present else None,
        # Read from the binary rather than hard-coded, so a re-freeze cannot
        # inherit a previous campaign's md5.
        "libReactantExtra_md5": _so_md5(),
        "libReactantExtra_path":
            "oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so",
        "machine": "AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, 1 NUMA node",
        "julia": "1.11.7",
        "batches": batches,
        "cross_batch_summary": summary,
    }
    # The destination is settable so a new campaign cannot overwrite a previous
    # one's frozen RAW file (which the paper's figure provenance cites by name).
    dest = os.environ.get(
        "PRELOWER_RAW_OUT", os.path.join(DIR, "advdiff_star_RAW_2026-08-25.json"))
    json.dump(out, open(dest, "w"), indent=2)
    print("\nwrote %s" % dest)

    if len(present) > 1:
        print("\n### cross-batch (%s)" % ", ".join(present))
        print("%-24s %-14s %26s %26s %20s"
              % ("Model", "Size", "Pre-Opt ms (min/med/max)",
                 "NoPreOpt ms (min/med/max)", "Speedup min/med/max"))
        for e in summary:
            if "speedup_across_batches" not in e:
                continue
            b, a = e["pre_opt_ms_across_batches"], e["no_pre_opt_ms_across_batches"]
            s = e["speedup_across_batches"]
            print("%-24s %-14s %8.1f/%8.1f/%8.1f %8.1f/%8.1f/%8.1f %5.1f/%5.1f/%5.1f"
                  % (e["model"], e["size"], b["min"], b["median"], b["max"],
                     a["min"], a["median"], a["max"], s["min"], s["median"], s["max"]))
