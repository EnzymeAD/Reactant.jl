#!/usr/bin/env python3
"""Build the raw JSON for the SICM ablation roster from one or more batch dirs.

For every (batch, cell) it locates the pplbench output dir from the cell log,
runs the postcheck summary (times, n_eff/Rhat, chain movement, cross-arm
agreement), and records the batch metadata (binary md5, loadavg before/after).

Usage: aggregate.py <out.json> <batch_dir> [<batch_dir> ...]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postcheck import summarize  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))

CELLS = ["car_spatial_scotland", "car_spatial_nyc", "lmm_radon", "hierarchical_mvn", "phylo_pois"]


def parse_meta(path):
    txt = open(path).read()
    meta = {"raw": txt, "cells": {}}
    for key in ["md5 at batch start", "md5 at batch end", "loadavg at batch start",
                "loadavg at batch end", "start", "end", "mtime", "nproc",
                "sibling taskset processes"]:
        m = re.search(r"^%s:\s*(.*)$" % re.escape(key), txt, re.M)
        if m:
            meta[key] = m.group(1).strip()
    for block in re.split(r"^--- cell ", txt, flags=re.M)[1:]:
        name = block.splitlines()[0].strip()
        cell = {}
        for key in ["start", "loadavg before", "md5 before", "end",
                    "loadavg after", "md5 after", "output dir", "script exit"]:
            m = re.search(r"^\s*%s:\s*(.*)$" % re.escape(key), block, re.M)
            if m:
                cell[key] = m.group(1).strip()
        meta["cells"][name] = cell
    return meta


def main():
    out_path = sys.argv[1]
    result = {"batches": {}}
    for bdir in sys.argv[2:]:
        tag = os.path.basename(os.path.normpath(bdir))
        meta = parse_meta(os.path.join(bdir, "batch_meta.txt"))
        entry = {"meta": {k: v for k, v in meta.items() if k != "raw"}, "cells": {}}
        for cell in CELLS:
            log = os.path.join(bdir, cell + ".log")
            if not os.path.exists(log):
                continue
            m = re.search(r"Output dir: (\S+)", open(log, errors="replace").read())
            if not m:
                entry["cells"][cell] = {"error": "no output dir in log"}
                continue
            run_dir = m.group(1).strip().rstrip("\r")
            if run_dir.startswith("./"):
                run_dir = os.path.join(BENCH, run_dir[2:])
            try:
                s = summarize(run_dir)
            except Exception as e:  # noqa: BLE001
                entry["cells"][cell] = {"error": "%s: %s" % (type(e).__name__, e),
                                        "run_dir": run_dir}
                continue
            s["batch_cell_meta"] = meta["cells"].get(cell, {})
            entry["cells"][cell] = s
        result["batches"][tag] = entry

    # cross-batch summary of the headline statistics
    summary = {}
    for cell in CELLS:
        rows = {}
        for tag, entry in result["batches"].items():
            c = entry["cells"].get(cell)
            if not c or "arms" not in c:
                continue
            arms = c["arms"]
            if "SICM" not in arms or "No SICM" not in arms:
                continue
            rows[tag] = {
                "sicm_last_s": arms["SICM"]["last_trial_s"],
                "nosicm_last_s": arms["No SICM"]["last_trial_s"],
                "speedup_last": arms["No SICM"]["last_trial_s"] / arms["SICM"]["last_trial_s"],
                "sicm_median_s": arms["SICM"]["trial_median_s"],
                "nosicm_median_s": arms["No SICM"]["trial_median_s"],
                "speedup_median": arms["No SICM"]["trial_median_s"] / arms["SICM"]["trial_median_s"],
                "sicm_trials": arms["SICM"]["all_trial_times_s"],
                "nosicm_trials": arms["No SICM"]["all_trial_times_s"],
                "run_dir": c["run_dir"],
            }
        summary[cell] = rows
    result["summary"] = summary

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out_path)
    for cell, rows in summary.items():
        for tag, r in rows.items():
            print("%-24s %-8s SICM last %8.3f med %8.3f | NoSICM last %9.3f med %9.3f | "
                  "%.1fx (last) %.1fx (median)"
                  % (cell, tag, r["sicm_last_s"], r["sicm_median_s"],
                     r["nosicm_last_s"], r["nosicm_median_s"],
                     r["speedup_last"], r["speedup_median"]))


if __name__ == "__main__":
    main()
