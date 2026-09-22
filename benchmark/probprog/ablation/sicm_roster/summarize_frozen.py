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
         "lmm_radon", "sar_columbus", "diamonds", "phylo_pois"]
LABEL = {
    "car_spatial_scotland": ("CholScale", "Lip Cancer CAR", "n=56"),
    "car_spatial_nyc": ("CholScale", "NYC Traffic CAR", "n=250"),
    "pgls": ("TriSolveScale", "Shorebird PGLS", "n=71"),
    "hierarchical_mvn": ("OuterProdScale", "Hier. MVN", "K=60, J=2"),
    "lmm_radon": ("EigenLift", "Radon LMM", "n=250"),
    "sar_columbus": ("DotScale", "Columbus SAR", "n=49"),
    "diamonds": ("GaussianSufficientStats", "Diamonds Regr.", "N=5000, K=25"),
    "phylo_pois": ("CholScale", "Phylo. Pois. Regr.", "n=200"),
}


# Problem-size sweep, 2026-09-10. Selected with FROZEN_ROWS=sweep so the
# default behaviour of this script (the frozen figure-10 rows) is untouched.
if os.environ.get("FROZEN_ROWS") == "sweep":
    # Only the NEW sizes. The frozen sizes (n=250, N=5000, K=60 J=2) are not
    # re-measured; their numbers come from the re-freeze RAW. Listing them here
    # would also be wrong: the per-key glob is a prefix match, so the key
    # "lmm_radon" would pick up "lmm_radon_n500_sicm.out.json".
    ORDER = ["car_spatial_nyc_n500", "car_spatial_nyc_n1000",
             "car_spatial_nyc_n1921",
             "lmm_radon_n500", "lmm_radon_n919",
             "diamonds_n20000", "diamonds_n53920",
             "hierarchical_mvn_K60_J5",
             "hierarchical_mvn_K120_J2", "hierarchical_mvn_K240_J2"]
    LABEL.update({
        "car_spatial_nyc_n500": ("CholScale", "NYC Traffic CAR", "n=500"),
        "car_spatial_nyc_n1000": ("CholScale", "NYC Traffic CAR", "n=1000"),
        "car_spatial_nyc_n1921": ("CholScale", "NYC Traffic CAR", "n=1921"),
        "lmm_radon_n500": ("EigenLift", "Radon LMM", "n=500"),
        "lmm_radon_n919": ("EigenLift", "Radon LMM", "n=919"),
        "diamonds_n20000": ("GaussianSufficientStats", "Diamonds Regr.", "N=20000"),
        "diamonds_n53920": ("GaussianSufficientStats", "Diamonds Regr.", "N=53920"),
        "hierarchical_mvn_K60_J5": ("OuterProdScale", "Hier. MVN", "K=60, J=5"),
        "hierarchical_mvn_K120_J2": ("OuterProdScale", "Hier. MVN", "K=120, J=2"),
        "hierarchical_mvn_K240_J2": ("OuterProdScale", "Hier. MVN", "K=240, J=2"),
    })


def geomean(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def per_draw_maxrel(a_obj, b_obj):
    """Per-draw maximum relative difference between the two arms.

    Same computation as arm_prefix.py, folded in so every row of the ledger
    carries its matched length instead of only a whole-window verdict. Returns
    (list of per-draw max relative differences, longest leading prefix under
    1e-8).
    """
    def load(o):
        out = {}
        for k, v in o.items():
            if k in ("compile_time", "run_time", "repeat_times_s"):
                continue
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            out[k] = arr
        return out

    A, B = load(a_obj), load(b_obj)
    keys = sorted(set(A) & set(B))
    if not keys:
        return [], 0
    n = min(min(A[k].shape[0] for k in keys), min(B[k].shape[0] for k in keys))
    per = []
    for i in range(n):
        worst = 0.0
        for k in keys:
            x, y = A[k][i], B[k][i]
            scale = max(float(np.max(np.abs(x))), 1.0)
            worst = max(worst, float(np.max(np.abs(x - y))) / scale)
        per.append(worst)
    pref = 0
    for v in per:
        if v < 1e-8:
            pref += 1
        else:
            break
    return per, pref


def draws(o):
    out = []
    for k in sorted(o):
        if k in ("compile_time", "run_time", "repeat_times_s"):
            continue
        out.append(np.asarray(o[k], dtype=float).ravel())
    return np.concatenate(out) if out else np.array([])


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = os.environ.get(
        "FROZEN_SUMMARY_OUT",
        os.path.join(HERE, "frozen_state_RAW_2026-08-25.json"))
    tags = argv or ["frozen1"]
    rows, raw = [], {}
    for key in ORDER:
        cell = {}
        suffix = ""
        for arm in ("sicm", "nosicm"):
            path = None
            for t in tags:
                # `<key>_<arm>.out.json`, or `<key>_seed<N>_<arm>.out.json` when
                # the row was re-run at a different seed to satisfy the
                # matched-trajectory gate. Newest wins.
                cands = sorted(glob.glob(os.path.join(LOGS, t, f"{key}*_{arm}.out.json")),
                               key=os.path.getmtime)
                if cands:
                    path = cands[-1]
            if path is None:
                continue
            base = os.path.basename(path)[:-len(f"_{arm}.out.json")]
            suffix = base[len(key):]
            o = json.load(open(path))
            ts = o.get("repeat_times_s")
            if not ts:
                continue
            cell[arm + "_obj"] = o
            cell[arm] = {
                "times": ts, "median": float(np.median(ts)),
                "geomean": geomean(ts), "min": float(np.min(ts)),
                "draws": draws(o), "path": path,
                "compile_time": o.get("compile_time"),
            }
        if len([k for k in cell if not k.endswith("_obj")]) != 2:
            print(f"  (skip {key}: {sorted(cell)} present)")
            continue
        a, b = cell["sicm"], cell["nosicm"]
        d = float(np.max(np.abs(a["draws"] - b["draws"])))
        scale = float(np.max(np.abs(a["draws"]))) or 1.0
        per_draw, prefix = per_draw_maxrel(cell["sicm_obj"], cell["nosicm_obj"])
        st = json.load(open(os.path.join(STATES, key + suffix + ".json")))
        rows.append({
            "key": key, "state_suffix": suffix,
            "pattern": LABEL[key][0], "label": LABEL[key][1],
            "size": LABEL[key][2],
            "sicm_median": a["median"], "nosicm_median": b["median"],
            "sicm_geomean": a["geomean"], "nosicm_geomean": b["geomean"],
            "ratio_median": b["median"] / a["median"],
            "ratio_geomean": b["geomean"] / a["geomean"],
            "arms_max_diff": d, "arms_rel_diff": d / scale,
            "arms_ok": bool(d / scale < 1e-8),
            # Matched length: how many leading draws the two arms agree on at
            # the ULP floor. A row that fails the whole-window gate is still
            # exactly matched over this prefix, which is the window its ratio
            # can honestly be quoted over.
            "n_draws": len(per_draw),
            "matched_prefix": prefix,
            "per_draw_max_rel_diff": per_draw,
            "adapted_step_size": st["adapted_step_size"],
            "initial_step_size": st["initial_step_size"],
            "inverse_mass_shape": st.get("inverse_mass_shape"),
            "sicm_times": a["times"], "nosicm_times": b["times"],
        })
        raw[key] = {"sicm": {k: v for k, v in a.items() if k != "draws"},
                    "nosicm": {k: v for k, v in b.items() if k != "draws"}}

    print(f"\n{'Pattern':<24}{'Model':<20}{'Size':<12}"
          f"{'SICM med':>10}{'OFF med':>10}{'ratio':>8}{'geo':>8}"
          f"{'arms rel':>11}  gate    matched")
    print("-" * 111)
    for r in rows:
        print(f"{r['pattern']:<24}{r['label']:<20}{r['size']:<12}"
              f"{r['sicm_median']:>10.4f}{r['nosicm_median']:>10.4f}"
              f"{r['ratio_median']:>7.2f}x{r['ratio_geomean']:>7.2f}x"
              f"{r['arms_rel_diff']:>11.2e}  {'PASS' if r['arms_ok'] else 'FAIL'}"
              f"  matched {r['matched_prefix']}/{r['n_draws']}")
    print("\nadapted step sizes (vs the fixed step the naive protocol used):")
    for r in rows:
        print(f"  {r['label']:<20} {r['initial_step_size']:.4g} -> "
              f"{r['adapted_step_size']:.4g} "
              f"({r['adapted_step_size']/r['initial_step_size']:.1f}x larger), "
              f"inverse mass {r['inverse_mass_shape']}")
    out = out_path
    json.dump({"protocol": "frozen-adapted-state, adaptation OFF both arms, "
                           "matched seeds, in-process repeats",
               "tags": tags, "rows": rows}, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
