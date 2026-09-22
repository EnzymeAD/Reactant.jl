#!/usr/bin/env python3
"""Machine-load adjudication for the re-freeze cells.

Reads every per-cell load sample file written by refreeze_cell.sh
(<logdir>/cells/<label>.samples.csv, columns epoch,load1,extcpu,our_running,
ambient_load1) or by the batch drivers (<logdir>/<tag>.samples.csv, columns
epoch,load1,extcpu) and reports, per cell:

  load1_max          raw 1-minute load average, peak over the cell window
  ambient_max        load1 minus this campaign's own runnable processes
                     (concurrent pinned cells on other CCDs are by design)
  ext_max            peak external (non-sbrantq) CPU percent, instantaneous
  ext_sustained      largest 5-minute rolling mean of external CPU
  frac_over          fraction of samples at or above the external-CPU limit

and two verdicts: STRICT (any instantaneous sample at or above the limit, or
ambient load1 above LOAD_MAX, is contamination) and SUSTAINED (only a 5-minute
mean at or above the limit counts). Both are reported for every cell.

Usage: adjudicate_cells.py <logdir> [<logdir> ...] [--limit 1000] [--load-max 24]
"""
import argparse
import csv
import glob
import json
import os


def stats(path, limit):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row or not row[0].isdigit():
                continue
            load1 = float(row[1]); ext = float(row[2])
            amb = float(row[4]) if len(row) > 4 else load1
            rows.append((int(row[0]), load1, ext, amb))
    if not rows:
        return None
    ext = [x[2] for x in rows]
    W = min(60, len(ext))
    run, best = 0.0, 0.0
    for i, v in enumerate(ext):
        run += v
        if i >= W:
            run -= ext[i - W]
        if i >= W - 1:
            best = max(best, run / W)
    return {
        "n_samples": len(rows),
        "t_start": rows[0][0], "t_end": rows[-1][0],
        "dur_s": rows[-1][0] - rows[0][0],
        "load1_max": max(x[1] for x in rows),
        "load1_mean": sum(x[1] for x in rows) / len(rows),
        "ambient_load1_max": max(x[3] for x in rows),
        "ext_max": max(ext),
        "ext_mean": sum(ext) / len(ext),
        "ext_sustained_5min": best,
        "frac_over_limit": sum(1 for v in ext if v >= limit) / len(ext),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", nargs="+")
    ap.add_argument("--limit", type=float, default=1000.0)
    ap.add_argument("--load-max", type=float, default=24.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    for d in a.logdir:
        for path in sorted(glob.glob(os.path.join(d, "cells", "*.samples.csv"))
                           + glob.glob(os.path.join(d, "*.samples.csv"))):
            label = os.path.basename(path)[: -len(".samples.csv")]
            s = stats(path, a.limit)
            if not s:
                continue
            s["dir"] = d
            s["verdict_strict"] = (
                "CLEAN" if s["ext_max"] < a.limit
                and s["ambient_load1_max"] <= a.load_max else "CONTAMINATED")
            s["verdict_sustained"] = (
                "CLEAN" if s["ext_sustained_5min"] < a.limit
                and s["ambient_load1_max"] <= a.load_max else "CONTAMINATED")
            gate = os.path.join(d, "cells", label + ".gate.txt")
            if os.path.isfile(gate):
                for line in open(gate):
                    for tok in line.split():
                        if tok.startswith("rc="):
                            s["rc"] = tok[3:]
                        if tok.startswith("verdict="):
                            s["driver_verdict"] = tok[len("verdict="):]
            out[f"{os.path.basename(d.rstrip('/'))}/{label}"] = s

    print(f"{'cell':<46}{'dur':>7}{'ld1':>7}{'amb':>7}{'extmax':>9}"
          f"{'extsus':>9}{'frac':>7}  strict / sustained")
    print("-" * 110)
    for k, s in sorted(out.items()):
        print(f"{k:<46}{s['dur_s']:>6}s{s['load1_max']:>7.1f}"
              f"{s['ambient_load1_max']:>7.1f}{s['ext_max']:>9.0f}"
              f"{s['ext_sustained_5min']:>9.0f}{s['frac_over_limit']:>7.1%}"
              f"  {s['verdict_strict']} / {s['verdict_sustained']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
