#!/usr/bin/env python3
"""
Extract benchmark results from pplbench log files into a clean CSV table.

Usage:
    python extract_results.py /tmp/bench_standard_all.log
    python extract_results.py /tmp/bench_standard_all.log --csv results.csv
    python extract_results.py /tmp/log1.log /tmp/log2.log  # merge multiple logs

Output columns:
    model, ppl, runtime_s, neff_median, neff_per_s
"""

import argparse
import csv
import re
import sys
from statistics import median


def parse_log(path):
    """Parse a pplbench log file. Returns list of result dicts."""
    results = []
    current_model = None
    current_ppl = None
    runtimes = []
    neff_values = []

    def flush():
        nonlocal current_ppl, runtimes, neff_values
        if current_ppl and runtimes:
            _save(results, current_model, current_ppl, runtimes, neff_values)
        current_ppl = None
        runtimes = []
        neff_values = []

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect model from PPLBench header — flush previous PPL first
        m = re.search(r"=== PPLBench: (.+?) ===", line)
        if m:
            flush()
            current_model = m.group(1).split("/")[-1]

        # Detect PPL — flush previous PPL first
        m = re.search(r"Starting inference on `(.+?)`", line)
        if m:
            flush()
            current_ppl = m.group(1)

        # Parse subprocess timing
        m = re.search(r"Subprocess compile=([\d.]+)s, run=([\d.]+)s", line)
        if m:
            runtimes.append(float(m.group(2)))

        # Parse n_eff median (next line has the values)
        m = re.search(r"overall n_eff \[min, median, max\]", line)
        if m and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            vals = re.findall(r"np\.float64\(([\d.e+-]+)\)", next_line)
            if len(vals) >= 2:
                neff_values.append(float(vals[1]))

        i += 1

    flush()
    return results


def _save(results, model, ppl, runtimes, neff_values):
    # Steady-state: exclude first trial
    steady = runtimes[1:] if len(runtimes) > 1 else runtimes
    med_runtime = median(steady) if steady else float("nan")
    med_neff = median(neff_values) if neff_values else float("nan")
    neff_per_s = med_neff / med_runtime if med_runtime > 0 else float("nan")
    results.append({
        "model": model or "unknown",
        "ppl": ppl,
        "runtime_s": round(med_runtime, 2),
        "neff_median": round(med_neff, 1),
        "neff_per_s": round(neff_per_s, 1),
    })


def print_table(results):
    """Print results as a formatted table grouped by model."""
    models = []
    seen = set()
    for r in results:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    ppls = []
    seen_ppl = set()
    for r in results:
        if r["ppl"] not in seen_ppl:
            ppls.append(r["ppl"])
            seen_ppl.add(r["ppl"])

    lookup = {}
    for r in results:
        lookup[(r["model"], r["ppl"])] = r

    model_w = max(len(m) for m in models)
    col_w = max(14, max(len(p) for p in ppls) + 2)

    header = f"{'Model':<{model_w}}"
    for p in ppls:
        header += f"  {p:>{col_w}}"
    print(header)
    print("-" * len(header))

    for model in models:
        row = f"{model:<{model_w}}"
        for ppl in ppls:
            r = lookup.get((model, ppl))
            if r:
                cell = f"{r['runtime_s']}s | {r['neff_per_s']:.0f}"
            else:
                cell = "-"
            row += f"  {cell:>{col_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Extract pplbench results")
    parser.add_argument("logs", nargs="+", help="Log file(s) from script -c or tee")
    parser.add_argument("--csv", help="Write CSV to this path")
    args = parser.parse_args()

    all_results = []
    for log in args.logs:
        all_results.extend(parse_log(log))

    if not all_results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print_table(all_results)

    if args.csv:
        fields = ["model", "ppl", "runtime_s", "neff_median", "neff_per_s"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
