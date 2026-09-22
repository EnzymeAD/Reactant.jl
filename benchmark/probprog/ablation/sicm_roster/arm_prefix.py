#!/usr/bin/env python3
"""Per-draw agreement between the two arms of a frozen-adapted-state cell.

Under the frozen protocol both arms start from one shared adapted state with
matched seeds and adaptation off, so they take the same trajectory. Bit
identity is impossible for the SICM ablation (the arms evaluate algebraically
equal but numerically different expressions), so the gate is the ULP floor. A
row fails the gate when a ULP difference flips a discrete sampler decision and
the two chains decorrelate from that draw on.

This reports, per draw, the maximum relative difference over all variables, and
the longest leading prefix of draws that stays under the threshold.

Usage: arm_prefix.py <sicm.out.json> <nosicm.out.json> [--tol 1e-8]
"""
import argparse
import json

import numpy as np

SKIP = {"compile_time", "run_time", "repeat_times_s"}


def load(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        if k in SKIP:
            continue
        a = np.asarray(v, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        out[k] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sicm")
    ap.add_argument("nosicm")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    A, B = load(a.sicm), load(a.nosicm)
    keys = sorted(set(A) & set(B))
    n = min(min(A[k].shape[0] for k in keys), min(B[k].shape[0] for k in keys))
    per_draw = []
    for i in range(n):
        worst = 0.0
        for k in keys:
            x, y = A[k][i], B[k][i]
            scale = max(float(np.max(np.abs(x))), 1.0)
            worst = max(worst, float(np.max(np.abs(x - y))) / scale)
        per_draw.append(worst)

    prefix = 0
    for v in per_draw:
        if v < a.tol:
            prefix += 1
        else:
            break

    print(f"draws compared: {n}   variables: {', '.join(keys)}")
    for i, v in enumerate(per_draw):
        print(f"  draw {i:>3}  max rel diff {v:.3e}{'   <-- first divergence' if i == prefix and prefix < n else ''}")
    print(f"\nlongest matched prefix at tol {a.tol:g}: {prefix} of {n} draws")
    print(f"whole-window max rel diff: {max(per_draw):.3e}")
    print(f"GATE: {'PASS' if prefix == n else 'FAIL'}")

    if a.json:
        json.dump({"n_draws": n, "tol": a.tol, "per_draw_max_rel_diff": per_draw,
                   "matched_prefix": prefix, "window_max_rel_diff": max(per_draw),
                   "pass": prefix == n, "sicm": a.sicm, "nosicm": a.nosicm},
                  open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
