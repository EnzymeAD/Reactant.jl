#!/usr/bin/env python3
"""Summarize a frozen-adapted-state batch: median and geomean per arm, the
arm ratio, and the matched-trajectory agreement gate.

Under this protocol both arms start from one shared adapted state with matched
seeds and adaptation off, so they take the SAME trajectory. Bit-identity is not
achievable for the SICM ablation (the arms evaluate algebraically equal but
numerically different expressions), so the gate is the ULP floor: relative
max |SICM - No SICM| over all draws must be tiny.

Usage: summarize_frozen.py <batch_tag> [more_tags...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs_frozen")
STATES = os.path.join(HERE, "adapted_states")

ORDER = ["car_spatial_scotland", "car_spatial_nyc", "pgls", "hierarchical_mvn",
         "lmm_radon", "sar_columbus", "diamonds"]
LABEL = {
    "car_spatial_scotland": ("CholScale", "Lip Cancer CAR", "n=56"),
    "car_spatial_nyc": ("CholScale", "NYC Traffic CAR", "n=250"),
    "pgls": ("TriSolveScale", "Shorebird PGLS", "n=71"),
    "hierarchical_mvn": ("OuterProdScale", "Hier. MVN", "K=60, J=2"),
    "lmm_radon": ("EigenLift", "Radon LMM", "n=250"),
    "sar_columbus": ("DotScale", "Columbus SAR", "n=49"),
    "diamonds": ("GaussianSufficientStats", "Diamonds Regr.", "N=5000, K=25"),
}


def geomean(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def draws(o):
    out = []
    for k in sorted(o):
        if k in ("compile_time", "run_time", "repeat_times_s"):
            continue
        out.append(np.asarray(o[k], dtype=float).ravel())
    return np.concatenate(out) if out else np.array([])


def main():
    tags = sys.argv[1:] or ["frozen1"]
    rows, raw = [], {}
    for key in ORDER:
        cell = {}
        for arm in ("sicm", "nosicm"):
            path = None
            for t in tags:
                p = os.path.join(LOGS, t, f"{key}_{arm}.out.json")
                if os.path.exists(p):
                    path = p
            if path is None:
                continue
            o = json.load(open(path))
            ts = o.get("repeat_times_s")
            if not ts:
                continue
            cell[arm] = {
                "times": ts, "median": float(np.median(ts)),
                "geomean": geomean(ts), "min": float(np.min(ts)),
                "draws": draws(o), "path": path,
                "compile_time": o.get("compile_time"),
            }
        if len(cell) != 2:
            print(f"  (skip {key}: {sorted(cell)} present)")
            continue
        a, b = cell["sicm"], cell["nosicm"]
        d = float(np.max(np.abs(a["draws"] - b["draws"])))
        scale = float(np.max(np.abs(a["draws"]))) or 1.0
        st = json.load(open(os.path.join(STATES, key + ".json")))
        rows.append({
            "key": key, "pattern": LABEL[key][0], "label": LABEL[key][1],
            "size": LABEL[key][2],
            "sicm_median": a["median"], "nosicm_median": b["median"],
            "sicm_geomean": a["geomean"], "nosicm_geomean": b["geomean"],
            "ratio_median": b["median"] / a["median"],
            "ratio_geomean": b["geomean"] / a["geomean"],
            "arms_max_diff": d, "arms_rel_diff": d / scale,
            "arms_ok": bool(d / scale < 1e-8),
            "adapted_step_size": st["adapted_step_size"],
            "initial_step_size": st["initial_step_size"],
            "inverse_mass_shape": st.get("inverse_mass_shape"),
            "sicm_times": a["times"], "nosicm_times": b["times"],
        })
        raw[key] = {"sicm": {k: v for k, v in a.items() if k != "draws"},
                    "nosicm": {k: v for k, v in b.items() if k != "draws"}}

    print(f"\n{'Pattern':<24}{'Model':<20}{'Size':<12}"
          f"{'SICM med':>10}{'OFF med':>10}{'ratio':>8}{'geo':>8}"
          f"{'arms rel':>11}  gate")
    print("-" * 111)
    for r in rows:
        print(f"{r['pattern']:<24}{r['label']:<20}{r['size']:<12}"
              f"{r['sicm_median']:>10.4f}{r['nosicm_median']:>10.4f}"
              f"{r['ratio_median']:>7.2f}x{r['ratio_geomean']:>7.2f}x"
              f"{r['arms_rel_diff']:>11.2e}  {'PASS' if r['arms_ok'] else 'FAIL'}")
    print("\nadapted step sizes (vs the fixed step the naive protocol used):")
    for r in rows:
        print(f"  {r['label']:<20} {r['initial_step_size']:.4g} -> "
              f"{r['adapted_step_size']:.4g} "
              f"({r['adapted_step_size']/r['initial_step_size']:.1f}x larger), "
              f"inverse mass {r['inverse_mass_shape']}")
    out = os.path.join(HERE, "frozen_state_RAW_2026-08-25.json")
    json.dump({"protocol": "frozen-adapted-state, adaptation OFF both arms, "
                           "matched seeds, in-process repeats",
               "tags": tags, "rows": rows}, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
