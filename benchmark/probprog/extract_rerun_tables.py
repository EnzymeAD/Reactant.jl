#!/usr/bin/env python3
"""Deterministic PPLBench sweep -> table extractor.

Reads structured PPLBench outputs (config.json + metrics.nc per run directory;
numbers are NEVER scraped from free-form logs) and emits:
  ledger.json  - machine-readable per-cell ledger
  tables.md    - GitHub-markdown tables
  tables.html  - self-contained HTML tables (artifact-ready building block)

Conventions (matching the 2026-08-02 cross_framework ledger):
  steady_wall_s_mean = mean of per-trial inference wall over trials 1..N-1
                       (trial 0 carries JIT/warm effects and is dropped)
  n_eff              = metrics.nc overall_neff [min, median, max]
  n_eff/s            = n_eff_median / steady_wall_s_mean
  headline replicate = the replicate whose steady wall is the median
  PLL                = mean over chains of the final-draw predictive LL

Anomaly flags (the machine is shared; bazel/ninja builds contaminate):
  - trial-wall spread (max-min)/median over trials 1..N-1 > --trial-spread
  - replicate steady-wall spread > --replicate-spread
  - max ambient load1 during the cell's run window > --load-threshold
    (requires the sweep dir's driver.log timestamps + load_samples.csv)
Each flagged cell gets its exact single-benchmark rerun command.

Usage:
  extract_rerun_tables.py --sweep logs/rerun_singlethread_<stamp>/ \
      [--sweep ...] [--outputs outputs/<ts> ...] [--out DIR] [--label TEXT]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import xarray as xr

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ORDER = [
    "logistic_regression", "robust_regression", "n_schools",
    "crowd_sourced_annotation", "noisy_or_topic", "gp_pois_regr_scaled",
]
SYSTEM_ORDER = ["Impulse", "NumPyro", "Stan", "Turing.jl"]
CONTINUOUS_NUTS = ["logistic_regression", "robust_regression", "n_schools"]
SVI_MODELS = ["logistic_regression", "robust_regression", "n_schools"]

SPAWN_RE = re.compile(
    r"^\[(?P<ts>[\d\- :]+)\] SPAWN (?P<label>\S+) \(core (?P<core>\d+)\): "
    r"run_pplbench\.sh (?P<args>.*)$")
START_RE = re.compile(
    r"^\[(?P<ts>[\d\- :]+)\] START (?P<label>\S+): run_pplbench\.sh (?P<args>.*)$")
END_RE = re.compile(r"^\[(?P<ts>[\d\- :]+)\] END\s+(?P<label>\S+): (?P<status>\S+)")
OUTDIR_RE = re.compile(r"^Output dir: (\./outputs/\S+)\s*$", re.M)


def parse_ts(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").timestamp()


def model_key(config):
    return config["model"]["class"].split(".")[0]


def load_sweep(sweep_dir):
    """Yield cell dicts {label, core, args, t_start, t_end, outdir} for a driver logdir."""
    driver_log = os.path.join(sweep_dir, "driver.log")
    cells = {}
    if os.path.isfile(driver_log):
        with open(driver_log) as f:
            for line in f:
                m = SPAWN_RE.match(line) or START_RE.match(line)
                if m:
                    d = m.groupdict()
                    cells[d["label"]] = {
                        "label": d["label"],
                        "core": d.get("core"),
                        "args": d["args"].strip(),
                        "t_start": parse_ts(d["ts"]),
                        "t_end": None,
                        "status": None,
                    }
                    continue
                m = END_RE.match(line)
                if m and m.group("label") in cells:
                    cells[m.group("label")]["t_end"] = parse_ts(m.group("ts"))
                    cells[m.group("label")]["status"] = m.group("status").rstrip(":")
    for label in sorted(cells):
        cell = cells[label]
        log_path = os.path.join(sweep_dir, label + ".log")
        cell["outdir"] = None
        if os.path.isfile(log_path):
            with open(log_path, errors="replace") as f:
                m = OUTDIR_RE.search(f.read())
            if m:
                cell["outdir"] = os.path.normpath(os.path.join(BENCH_DIR, m.group(1)))
        yield cell


def load_load_samples(sweep_dir):
    path = os.path.join(sweep_dir, "load_samples.csv")
    if not os.path.isfile(path):
        return []
    samples = []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                samples.append((int(parts[0]), float(parts[1])))
    return samples


def max_load_in_window(samples, t0, t1):
    vals = [l for (t, l) in samples if t0 <= t <= t1]
    return max(vals) if vals else None


def read_output_dir(outdir):
    """Return list of per-ppl cell measurement dicts from one PPLBench output dir."""
    cfg_path = os.path.join(outdir, "config.json")
    nc_path = os.path.join(outdir, "metrics.nc")
    if not (os.path.isfile(cfg_path) and os.path.isfile(nc_path)):
        return []
    config = json.load(open(cfg_path))
    mkey = model_key(config)
    legend_to_mode = {}
    for p in config.get("ppls", []):
        legend_to_mode[p["legend"]["name"]] = (
            "svi" if "SVI" in p["inference"]["class"] else "nuts")
    out = []
    with xr.open_dataset(nc_path) as m:
        for i, legend in enumerate(np.atleast_1d(m.ppl.values)):
            legend = str(legend)
            timing = m.timing.isel(ppl=i).values  # (chain, phase[compile, infer])
            phases = [str(p) for p in m.phase.values]
            infer_idx = phases.index("infer")
            compile_idx = phases.index("compile") if "compile" in phases else None
            walls = [float(w) for w in timing[:, infer_idx]]
            steady = walls[1:] if len(walls) > 1 else walls
            steady_mean = float(np.mean(steady))
            pct = [str(p) for p in m.percentile.values]
            neff = m.overall_neff.isel(ppl=i).values
            neff_med = float(neff[pct.index("median")])
            stored_nps = m.overall_neff_per_time.isel(ppl=i).values
            pll_final = [float(x) for x in m.pll.isel(ppl=i, draw=-1).values]
            out.append({
                "model": mkey,
                "model_args": config["model"].get("args", {}),
                "legend": legend,
                "mode": legend_to_mode.get(legend, "nuts"),
                "iterations": config.get("iterations"),
                "num_warmup": config.get("num_warmup"),
                "outdir": outdir,
                "compile_s": (float(timing[0, compile_idx])
                              if compile_idx is not None else None),
                "per_trial_wall_s": walls,
                "steady_wall_s_mean": steady_mean,
                "trial_wall_spread": (
                    (max(steady) - min(steady)) / float(np.median(steady))
                    if len(steady) > 1 and np.median(steady) > 0 else 0.0),
                "n_eff": {"min": float(neff[pct.index("min")]),
                          "median": neff_med,
                          "max": float(neff[pct.index("max")])},
                "n_eff_per_s": (neff_med / steady_mean) if steady_mean > 0 else None,
                "n_eff_per_s_stored_median": float(stored_nps[pct.index("median")]),
                "per_trial_final_pll": pll_final,
                "final_pll_mean": float(np.mean(pll_final)),
            })
    return out


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def order_key(seq):
    def k(x):
        return (seq.index(x), "") if x in seq else (len(seq), x)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--outputs", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--trial-spread", type=float, default=0.25)
    ap.add_argument("--replicate-spread", type=float, default=0.15)
    ap.add_argument("--load-threshold", type=float, default=24.0)
    args = ap.parse_args()

    measurements = []  # (cell_meta_or_None, measurement)
    for sweep in args.sweep:
        sweep = os.path.abspath(sweep)
        samples = load_load_samples(sweep)
        for cell in load_sweep(sweep):
            if not cell["outdir"]:
                measurements.append((dict(cell, sweep=sweep, max_load1=None), None))
                continue
            maxload = (max_load_in_window(samples, cell["t_start"], cell["t_end"])
                       if cell["t_end"] else None)
            for meas in read_output_dir(cell["outdir"]):
                measurements.append(
                    (dict(cell, sweep=sweep, max_load1=maxload), meas))
    for outdir in args.outputs:
        for meas in read_output_dir(os.path.abspath(outdir)):
            measurements.append((None, meas))

    # Group replicates by (mode, model, legend)
    groups = defaultdict(list)
    missing = []
    for cell, meas in measurements:
        if meas is None:
            missing.append(cell)
            continue
        groups[(meas["mode"], meas["model"], meas["legend"])].append((cell, meas))

    ledger = {"label": args.label, "cells": {}, "flags": [], "missing_cells": [
        {"label": c["label"], "sweep": c.get("sweep"), "status": c.get("status")}
        for c in missing]}
    for key in sorted(groups):
        mode, model, legend = key
        reps = groups[key]
        walls = [m["steady_wall_s_mean"] for _, m in reps]
        # median-wall replicate is the headline; for even counts, the lower median
        reps_by_wall = sorted(range(len(reps)), key=lambda i: walls[i])
        headline_idx = reps_by_wall[(len(reps) - 1) // 2]
        cell, headline = reps[headline_idx]
        rep_spread = ((max(walls) - min(walls)) / float(np.median(walls))
                      if len(walls) > 1 and np.median(walls) > 0 else 0.0)
        entry = {
            "headline": headline,
            "replicates": [m for _, m in reps],
            "replicate_spread": rep_spread,
            "n_replicates": len(reps),
            "max_load1": cell.get("max_load1") if cell else None,
            "core": cell.get("core") if cell else None,
            "rerun_cmd": (
                f"PPLBENCH_TASKSET_CPUS={cell['core']} ./run_pplbench.sh {cell['args']}"
                if cell and cell.get("core") is not None else None),
        }
        flags = []
        if headline["trial_wall_spread"] > args.trial_spread:
            flags.append(f"trial-wall spread {headline['trial_wall_spread']:.0%}")
        if rep_spread > args.replicate_spread:
            flags.append(f"replicate spread {rep_spread:.0%}")
        if entry["max_load1"] is not None and entry["max_load1"] > args.load_threshold:
            flags.append(f"ambient load1 up to {entry['max_load1']:.0f} during run")
        entry["flags"] = flags
        if flags:
            ledger["flags"].append({
                "cell": f"{mode}/{model}/{legend}", "flags": flags,
                "rerun_cmd": entry["rerun_cmd"]})
        ledger["cells"][f"{mode}/{model}/{legend}"] = entry

    # ------------------------------------------------------------- tables
    def cells_for(mode):
        rows = defaultdict(dict)
        for k, e in ledger["cells"].items():
            kmode, model, legend = k.split("/", 2)
            if kmode == mode:
                rows[model][legend] = e
        return rows

    def fmt(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"

    md, html_rows = [], []
    md.append(f"# Benchmark tables{(' - ' + args.label) if args.label else ''}\n")
    md.append("Protocol: single-threaded, taskset-pinned (cross-system rows); "
              "steady wall = mean over trials 1..N-1; headline = median-wall "
              "replicate; n_eff/s = median n_eff / steady wall.\n")

    summaries = {}
    for mode, title in [("nuts", "NUTS"), ("svi", "SVI")]:
        rows = cells_for(mode)
        if not rows:
            continue
        md.append(f"\n## {title}\n")
        systems = sorted({s for r in rows.values() for s in r},
                         key=order_key(SYSTEM_ORDER))
        header = ["model"] + [f"{s} n_eff/s" for s in systems] + \
                 [f"{s} wall(s)" for s in systems] + [f"{s} PLL" for s in systems]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for model in sorted(rows, key=order_key(MODEL_ORDER)):
            r = rows[model]
            line = [model]
            for s in systems:
                h = r.get(s, {}).get("headline")
                flag = " *" if r.get(s, {}).get("flags") else ""
                line.append((fmt(h["n_eff_per_s"], 2) + flag) if h else "-")
            for s in systems:
                h = r.get(s, {}).get("headline")
                line.append(fmt(h["steady_wall_s_mean"]) if h else "-")
            for s in systems:
                h = r.get(s, {}).get("headline")
                line.append(fmt(h["final_pll_mean"], 2) if h else "-")
            md.append("| " + " | ".join(line) + " |")

        # geomean ratios vs Impulse
        base_models = CONTINUOUS_NUTS if mode == "nuts" else SVI_MODELS
        ratios = {}
        for s in systems:
            if s == "Impulse":
                continue
            rs = []
            for model in base_models:
                hi = rows.get(model, {}).get("Impulse", {}).get("headline")
                hb = rows.get(model, {}).get(s, {}).get("headline")
                if hi and hb and hb["n_eff_per_s"]:
                    rs.append(hi["n_eff_per_s"] / hb["n_eff_per_s"])
            if len(rs) == len(base_models):
                ratios[s] = geomean(rs)
        if ratios:
            summaries[mode] = ratios
            md.append("")
            md.append(f"Geomean Impulse speedup over ({', '.join(base_models)}): "
                      + "; ".join(f"{s}: {v:.2f}x" for s, v in sorted(
                          ratios.items(), key=lambda kv: order_key(SYSTEM_ORDER)(kv[0]))))

    if ledger["flags"]:
        md.append("\n## Anomaly flags (rerun candidates)\n")
        for f in ledger["flags"]:
            md.append(f"- `{f['cell']}`: {', '.join(f['flags'])}")
            if f["rerun_cmd"]:
                md.append(f"  - rerun: `{f['rerun_cmd']}`")
    if ledger["missing_cells"]:
        md.append("\n## Cells with no output dir (failed or still running)\n")
        for c in ledger["missing_cells"]:
            md.append(f"- {c['label']} (status: {c['status']})")

    md_text = "\n".join(md) + "\n"

    # minimal, self-contained html (tables only; artifact styling comes later)
    import html as _html
    html_parts = ["<div class='bench-tables'>"]
    in_table = False
    for line in md_text.splitlines():
        if line.startswith("| ") or line.startswith("|-"):
            cellsep = [c.strip() for c in line.strip("|").split("|")]
            if line.startswith("|-"):
                continue
            if not in_table:
                html_parts.append("<table><thead><tr>" + "".join(
                    f"<th>{_html.escape(c)}</th>" for c in cellsep)
                    + "</tr></thead><tbody>")
                in_table = True
            else:
                html_parts.append("<tr>" + "".join(
                    f"<td>{_html.escape(c)}</td>" for c in cellsep) + "</tr>")
        else:
            if in_table:
                html_parts.append("</tbody></table>")
                in_table = False
            if line.startswith("# "):
                html_parts.append(f"<h1>{_html.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{_html.escape(line[3:])}</h2>")
            elif line.startswith("- "):
                html_parts.append(f"<p>{_html.escape(line)}</p>")
            elif line.strip():
                html_parts.append(f"<p>{_html.escape(line)}</p>")
    if in_table:
        html_parts.append("</tbody></table>")
    html_parts.append("</div>")
    html_text = "\n".join(html_parts) + "\n"

    outdir = args.out or (os.path.join(args.sweep[0], "tables")
                          if args.sweep else os.path.join(BENCH_DIR, "tables"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ledger.json"), "w") as f:
        json.dump(ledger, f, indent=1, sort_keys=True)
    with open(os.path.join(outdir, "tables.md"), "w") as f:
        f.write(md_text)
    with open(os.path.join(outdir, "tables.html"), "w") as f:
        f.write(html_text)
    print(md_text)
    print(f"[wrote {outdir}/ledger.json, tables.md, tables.html]", file=sys.stderr)


if __name__ == "__main__":
    main()
