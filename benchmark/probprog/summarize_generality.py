#!/usr/bin/env python3
"""Summarize the pinned generality (MH throughput) pass.

Reads the per-cell transcripts written by run_generality_pinned.sh, extracts
every repetition line

    <label> REP wall <seconds> s | per-move <microseconds> us

(one per timed repetition; the harnesses discard one warm-up repetition before
the timed loop), and reports the MEDIAN per-move microseconds and the derived
moves per second for each label, together with the min (the statistic the
2026-08-21 pass used) so the two are comparable. Also collects the harnesses'
validation lines and each cell's load gate.

Usage: summarize_generality.py <logdir> [--json OUT]
"""
import argparse
import json
import os
import re
import statistics
import sys

REP = re.compile(r"^(?P<label>.+?) REP wall (?P<wall>[0-9.]+) s \| per-move (?P<us>[0-9.]+) us")
VALID = re.compile(
    r"(per-move .* \| .*=|per-move .*us \| arm =|MAP s =|P\(m=line\) =|flags =|slope=|accepts|"
    r"posterior sanity|bench   :|assess=|\|Δtau\||cholesky|chol in-while)")


def strip_cr(s):
    return s.replace("\r", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cells = {}
    for fn in sorted(os.listdir(a.logdir)):
        if not fn.endswith(".log"):
            continue
        path = os.path.join(a.logdir, fn)
        cell = fn[:-4]
        gate = {}
        gpath = os.path.join(a.logdir, "cells", cell + ".gate.txt")
        if os.path.isfile(gpath):
            for line in open(gpath):
                for tok in line.strip().split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        gate[k] = v
        reps, valid = {}, []
        with open(path, errors="replace") as f:
            for line in f:
                line = strip_cr(line).rstrip("\n")
                m = REP.match(line.strip())
                if m:
                    reps.setdefault(m.group("label").strip(), []).append(
                        (float(m.group("wall")), float(m.group("us"))))
                    continue
                if VALID.search(line) and "REP wall" not in line:
                    valid.append(line.strip())
        arms = {}
        for label, vals in reps.items():
            us = [u for _, u in vals]
            walls = [w for w, _ in vals]
            arms[label] = {
                "n_reps": len(us),
                "per_move_us_median": statistics.median(us),
                "per_move_us_min": min(us),
                "per_move_us_max": max(us),
                "per_move_us_all": us,
                "wall_s_all": walls,
                "spread_max_min_over_median": (
                    (max(us) - min(us)) / statistics.median(us) if us else None),
                # robust spread: the middle half of the repetitions. The
                # max-min spread is dominated by the occasional multi-ms
                # outlier (a GC pause or a scheduler hiccup) and says nothing
                # about how well the median is determined.
                "iqr_over_median": (
                    (statistics.quantiles(us, n=4)[2] - statistics.quantiles(us, n=4)[0])
                    / statistics.median(us) if len(us) >= 4 else None),
                "moves_per_s_median": 1e6 / statistics.median(us) if us else None,
            }
        cells[cell] = {"gate": gate, "arms": arms, "validation_lines": valid}

    for cell, d in cells.items():
        if not d["arms"]:
            continue
        print(f"== {cell}   [{d['gate'].get('verdict','?')} "
              f"load1_max={d['gate'].get('load1_max','?')} "
              f"ambient={d['gate'].get('ambient_load1_max','?')} "
              f"ext_max={d['gate'].get('extcpu_max','?')} "
              f"dur={d['gate'].get('dur_s','?')}s]")
        for label, s in d["arms"].items():
            print(f"   {label:28s} median {s['per_move_us_median']:9.4f} us "
                  f"(min {s['per_move_us_min']:9.4f}, n={s['n_reps']}, "
                  f"iqr {s['iqr_over_median']:.1%}, "
                  f"max-min {s['spread_max_min_over_median']:.0%})  "
                  f"= {s['moves_per_s_median']:,.0f} moves/s")
        for v in d["validation_lines"]:
            print(f"     | {v}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(cells, f, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
