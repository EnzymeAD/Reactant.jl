#!/usr/bin/env python3
"""Aggregate a pre-lowering controlled batch into a raw-timing JSON plus a
verification report (cross-arm accept identity, sample agreement, syevd)."""
import json, os, re, sys, statistics

DIR = "/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog/ablation/pre_lowering"

ROWS = [
    ("Heat Equation",         "heat",      "D=100, T=100", "heat_s1_B",     "heat_s1_A",     1),
    ("Heat Equation (large)", "heat",      "D=300, T=100", "heat_s3_B",     "heat_s3_A",     3),
    ("Cubic Kernel",          "cubic",     "D=60, T=50",   "cubic_small_B", "cubic_small_A", None),
    ("Cubic Kernel (large)",  "cubic",     "D=100, T=100", "cubic_large_B", "cubic_large_A", None),
    ("Projected Dynamics",    "projected", "D=100, T=100", "projected_B",   "projected_A",   None),
]
CELL_JSON = {
    "cubic_small_B": "cubic_small_armB.json", "cubic_small_A": "cubic_small_armA.json",
    "cubic_large_B": "cubic_large_armB.json", "cubic_large_A": "cubic_large_armA.json",
    "projected_B": "projected_armlarge_armB.json", "projected_A": "projected_armlarge_armA.json",
}
ALL_RE = re.compile(r"All:\s*\[([^\]]*)\]ms")
MED_RE = re.compile(r"Median \(20 NUTS samples\):\s*([0-9.]+)ms")
COMP_RE = re.compile(r"Compile:\s*([0-9.]+)s")
IR_RE = re.compile(r"^\s+(\S+)\s+in=(\d+)\s+out=(\d+)\s+total=(\d+)\s*$")


def parse_log(logs, tag):
    txt = open(os.path.join(logs, tag + ".log"), errors="replace").read()
    m = ALL_RE.search(txt)
    d = {"log": tag + ".log",
         "times_ms": [float(x) for x in m.group(1).split(",")] if m else None,
         "median_ms": float(MED_RE.search(txt).group(1)) if MED_RE.search(txt) else None,
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
        d.update({k: j[k] for k in ("samples", "accepted", "diverging",
                                    "log_densities", "verify_seed")})
        d["times_ms"] = j["times_ms"]; d["median_ms"] = j["median_ms"]
        d["compile_s"] = j["compile_s"]
    return d


def run(logs, out_json, label):
    out = {"batch": label,
           "meta": open(os.path.join(logs, "batch_meta.txt"), errors="replace").read(),
           "rows": []}
    print("\n### %s" % label)
    print("%-24s %-14s %10s %12s %9s %8s %8s" %
          ("Model", "Size", "Pre-Opt", "No Pre-Opt", "Speedup", "sprB", "sprA"))
    for label_, model, size, tb, ta, scale in ROWS:
        b, a = load_cell(logs, tb), load_cell(logs, ta)
        sp = a["median_ms"] / b["median_ms"]
        ver = {"syevd_A": a["ir"].get("syevd", {}).get("total"),
               "syevd_B": b["ir"].get("syevd", {}).get("total")}
        if a.get("samples"):
            ver["accept_identical"] = (a["accepted"] == b["accepted"])
            ver["diverging_identical"] = (a["diverging"] == b["diverging"])
            ver["n_accepted"] = sum(a["accepted"])
            ver["max_abs_sample_diff"] = max(
                abs(x - y) for ra, rb in zip(a["samples"], b["samples"])
                for x, y in zip(ra, rb))
            ver["max_abs_logdensity_diff"] = max(
                abs(x - y) for x, y in zip(a["log_densities"], b["log_densities"]))
            ver["verify_seed"] = a["verify_seed"]
        elif scale:
            pp = {}
            for arm in "AB":
                fp = os.path.join(logs, "parity", "heat_s%d_arm%s.json" % (scale, arm))
                pp[arm] = json.load(open(fp)) if os.path.exists(fp) else None
            if pp["A"] and pp["B"]:
                ver["accept_identical"] = (pp["A"]["accepted"] == pp["B"]["accepted"])
                ver["n_accepted"] = sum(pp["A"]["accepted"][:20])
                ver["max_abs_sample_diff"] = max(
                    abs(x - y) for ra, rb in zip(pp["A"]["samples"], pp["B"]["samples"])
                    for x, y in zip(ra, rb))
                ver["verify_seed"] = pp["A"]["seed"]
        sprb = (max(b["times_ms"]) - min(b["times_ms"])) / b["median_ms"]
        spra = (max(a["times_ms"]) - min(a["times_ms"])) / a["median_ms"]
        out["rows"].append({"model": label_, "size": size, "speedup": sp,
                            "pre_opt_B": b, "no_pre_opt_A": a, "verification": ver,
                            "spread_B_frac": sprb, "spread_A_frac": spra})
        print("%-24s %-14s %10.1f %12.1f %8.1fx %7.0f%% %7.0f%%  %s" %
              (label_, size, b["median_ms"], a["median_ms"], sp,
               100 * sprb, 100 * spra,
               json.dumps({k: ver[k] for k in ("accept_identical",
                                               "max_abs_sample_diff",
                                               "syevd_A", "syevd_B") if k in ver})))
    if out_json:
        json.dump(out, open(out_json, "w"), indent=2)
    return out


if __name__ == "__main__":
    logs = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    run(logs, out_json, sys.argv[3] if len(sys.argv) > 3 else logs)
