#!/usr/bin/env python3
"""Turn the 2026-09-10 problem-size sweep into SWEEP.md.

Reads
  * the sweep's own figure-10 summary (summarize_frozen.py FROZEN_ROWS=sweep)
  * the sweep's own figure-9 summary (aggregate_v2.py AGG_ROWS=sweep)
  * the 2026-09-08 re-freeze RAW files, for the sizes that are NOT re-measured
    (they were measured on this binary under this protocol two days earlier)
  * every accepted cell's 5-second load record (<tag>.samples.csv), which is
    what the per-cell load verdict and cell duration come from. The roster
    driver truncates batch_meta.txt at every stage invocation, so the CSVs are
    the authoritative per-cell load evidence.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROSTER = os.path.normpath(os.path.join(HERE, ".."))
PRE = os.path.normpath(os.path.join(ROSTER, "..", "pre_lowering"))
BENCH = os.path.normpath(os.path.join(ROSTER, "..", ".."))
EV = "/mnt4/sbrantq/probprog/oopsla/evidence"
TAG = "size_sweep_2026-09-10"
RLOG = os.path.join(ROSTER, "logs_frozen", TAG)
PLOG = os.path.join(PRE, "logs")
LIMIT = 1000.0


def load_record(path):
    if not os.path.exists(path):
        return None
    ext, l1, ep = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ext.append(float(row["extcpu"])); l1.append(float(row["load1"]))
                ep.append(float(row["epoch"]))
            except (ValueError, KeyError, TypeError):
                continue
    if not ext:
        return None
    over = sum(1 for e in ext if e >= LIMIT)
    return {"ext_max": max(ext), "ext_median": sorted(ext)[len(ext) // 2],
            "load1_max": max(l1), "n_samples": len(ext), "n_over": over,
            "duration_s": (max(ep) - min(ep)) if len(ep) > 1 else 0.0,
            "verdict": "PASS" if over == 0 else "FAIL"}


def fmt_load(r):
    if not r:
        return "no record"
    return ("ext max %.0f%% (median %.0f%%), %d/%d samples >= %d%%, load1 max %.1f, %s"
            % (r["ext_max"], r["ext_median"], r["n_over"], r["n_samples"],
               LIMIT, r["load1_max"], r["verdict"]))


def mmss(s):
    if not s:
        return "-"
    return "%d:%02d" % (int(s) // 60, int(s) % 60)


def fig10_rows():
    """(pattern, label, size, sicm_med, nosicm_med, ratio_med, ratio_geo,
        matched, n_draws, agreement, step, imm, sicm_csv, nosicm_csv, source)"""
    out = []
    froz = json.load(open(os.path.join(
        EV, "refreeze_2026-09-08", "fig10_sicm", "frozen_state_RAW.json")))
    fz = {r["key"]: r for r in froz["rows"]}
    swp_path = os.path.join(HERE, "frozen_state_RAW_sweep.json")
    sw = {}
    if os.path.exists(swp_path):
        sw = {r["key"]: r for r in json.load(open(swp_path))["rows"]}
    order = [
        ("car_spatial_nyc", "refreeze"), ("car_spatial_nyc_n500", "sweep"),
        ("car_spatial_nyc_n1000", "sweep"), ("car_spatial_nyc_n1921", "sweep"),
        ("lmm_radon", "refreeze"), ("lmm_radon_n500", "sweep"),
        ("lmm_radon_n919", "sweep"),
        ("diamonds", "refreeze"), ("diamonds_n20000", "sweep"),
        ("diamonds_n53920", "sweep"),
        ("hierarchical_mvn", "refreeze"), ("hierarchical_mvn_K60_J5", "sweep"),
        ("hierarchical_mvn_K120_J2", "sweep"),
        ("hierarchical_mvn_K240_J2", "sweep"),
        ("car_spatial_scotland", "refreeze"), ("pgls", "refreeze"),
        ("phylo_pois", "refreeze"),
    ]
    for key, src in order:
        r = (fz if src == "refreeze" else sw).get(key)
        if r is None:
            continue
        rec = {k: r[k] for k in
               ("pattern", "label", "size", "sicm_median", "nosicm_median",
                "sicm_geomean", "nosicm_geomean", "ratio_median",
                "ratio_geomean", "matched_prefix", "n_draws", "arms_rel_diff",
                "adapted_step_size", "inverse_mass_shape")}
        rec["key"] = key
        rec["source"] = src
        if src == "sweep":
            rec["load_sicm"] = load_record(os.path.join(RLOG, key + "_sicm.samples.csv"))
            rec["load_nosicm"] = load_record(os.path.join(RLOG, key + "_nosicm.samples.csv"))
        else:
            rec["load_sicm"] = rec["load_nosicm"] = None
        out.append(rec)
    return out


def fig09_rows():
    # aggregate_v2.py nests its rows under the batch name it was run on.
    def batch_rows(d):
        return d["batches"][d["headline_batch"]]["rows"]
    froz = json.load(open(os.path.join(
        EV, "refreeze_2026-09-08", "fig09_prelowering", "RAW.json")))
    fz = {(r["model"], r["size"]): r for r in batch_rows(froz)}
    swp_path = os.path.join(HERE, "prelowering_RAW_sweep.json")
    sw = {}
    if os.path.exists(swp_path):
        for r in batch_rows(json.load(open(swp_path))):
            sw[(r["model"], r["size"])] = r
    order = [
        (("Heat Equation", "D=100, T=100"), "refreeze", None, None),
        (("Heat Equation (large)", "D=300, T=100"), "refreeze", None, None),
        (("Heat Equation T=400", "D=300, T=400"), "sweep", "heat_s7_B", "heat_s7_A"),
        (("Heat Equation D=1000", "D=1000, T=100"), "sweep", "heat_s6_B", "heat_s6_A"),
        (("Advection-Diffusion", "D=200, T=100"), "refreeze", None, None),
        (("Adv-Diff D=500", "D=500, T=250"), "sweep", "advdiff_s3_B", "advdiff_s3_A"),
        (("Adv-Diff D=1000", "D=1000, T=500"), "sweep", "advdiff_s4_B", "advdiff_s4_A"),
        (("STAR (ILINet)", "N=47, T=104"), "refreeze", None, None),
    ]
    out = []
    for k, src, tb, ta in order:
        r = (fz if src == "refreeze" else sw).get(k)
        if r is None or r.get("failed"):
            continue
        v = r.get("verification", {})
        rec = {"model": k[0], "size": k[1], "source": src,
               "b_med": r["pre_opt_B"]["median_ms"],
               "b_geo": r["pre_opt_B"]["geomean_ms"],
               "a_med": r["no_pre_opt_A"]["median_ms"],
               "a_geo": r["no_pre_opt_A"]["geomean_ms"],
               "ratio": r["speedup"], "ratio_geo": r["speedup_geomean"],
               "accept_identical": v.get("accept_identical"),
               "diverging_identical": v.get("diverging_identical"),
               "n_accepted": v.get("n_accepted"),
               "n_diverging": v.get("n_diverging"),
               "max_abs_sample_diff": v.get("max_abs_sample_diff"),
               "syevd_A": v.get("syevd_A"), "syevd_B": v.get("syevd_B"),
               "step": (r.get("adapted_state") or {}).get("adapted_step_size"),
               "imm_rank": (r.get("adapted_state") or {}).get("inverse_mass_rank"),
               "ir_B": r["pre_opt_B"].get("ir", {}),
               "ir_A": r["no_pre_opt_A"].get("ir", {}),
               "compile_B": r["pre_opt_B"].get("compile_s"),
               "compile_A": r["no_pre_opt_A"].get("compile_s")}
        rec["load_B"] = load_record(os.path.join(PLOG, tb + ".samples.csv")) if tb else None
        rec["load_A"] = load_record(os.path.join(PLOG, ta + ".samples.csv")) if ta else None
        out.append(rec)
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        EV, TAG, "SWEEP.md")
    f10, f09 = fig10_rows(), fig09_rows()
    L = []
    A = L.append
    A("# Problem-size sweep, 2026-09-10: per-cell table\n")
    A("One binary (`libReactantExtra.so` md5 `95aeb90de165d03c09db5736d67951cc`), "
      "the protocol and gates of the 2026-09-08 re-freeze. Rows marked "
      "`re-freeze` are the already-frozen sizes, measured on this same binary "
      "under this same protocol on 2026-09-08 and not re-run; rows marked "
      "`sweep` were measured on 2026-09-10. Cell duration is the accepted "
      "attempt's wall clock, compilation included (the timings exclude it).\n")

    A("\n## Figure 10, SICM whole-pass ablation\n")
    A("| row | size | SICM (s) | no SICM (s) | ratio median | ratio geomean | "
      "matched | agreement | adapted step | inv mass | cell duration (SICM / no SICM) | load verdict | measured |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in f10:
        imm = "x".join(str(x) for x in (r["inverse_mass_shape"] or []))
        A("| %s (%s) | %s | %.5f | %.5f | **%.2fx** | %.2fx | %d/%d | %.2e | %.4g | %s | %s / %s | %s | %s |"
          % (r["label"], r["pattern"], r["size"], r["sicm_median"],
             r["nosicm_median"], r["ratio_median"], r["ratio_geomean"],
             r["matched_prefix"], r["n_draws"], r["arms_rel_diff"],
             r["adapted_step_size"], imm,
             mmss((r["load_sicm"] or {}).get("duration_s")),
             mmss((r["load_nosicm"] or {}).get("duration_s")),
             ((r["load_sicm"] or {}).get("verdict", "-") + " / " +
              (r["load_nosicm"] or {}).get("verdict", "-")) if r["load_sicm"]
             else "re-freeze ledger", r["source"]))
    A("\nPer-cell load records (sweep cells only):\n")
    for r in f10:
        if r["source"] != "sweep":
            continue
        A("- `%s` SICM: %s" % (r["key"], fmt_load(r["load_sicm"])))
        A("- `%s` no SICM: %s" % (r["key"], fmt_load(r["load_nosicm"])))

    A("\n## Figure 9, standard optimization passes on probabilistic IR\n")
    A("| row | size | with std. opt. (ms) | without (ms) | ratio median | "
      "ratio geomean | accepts identical | diverges identical | max abs sample diff | "
      "syevd A/B | adapted step | cell duration (with / without) | load verdict | measured |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in f09:
        A("| %s | %s | %.1f | %.1f | **%.2fx** | %.2fx | %s | %s | %.3e | %s/%s | %.6g | %s / %s | %s | %s |"
          % (r["model"], r["size"], r["b_med"], r["a_med"], r["ratio"],
             r["ratio_geo"], r["accept_identical"], r["diverging_identical"],
             r["max_abs_sample_diff"] if r["max_abs_sample_diff"] is not None else float("nan"),
             r["syevd_A"], r["syevd_B"], r["step"] or float("nan"),
             mmss((r["load_B"] or {}).get("duration_s")),
             mmss((r["load_A"] or {}).get("duration_s")),
             ((r["load_B"] or {}).get("verdict", "-") + " / " +
              (r["load_A"] or {}).get("verdict", "-")) if r["load_B"]
             else "re-freeze ledger", r["source"]))
    A("\nIn-loop LAPACK, from each cell's own IR report "
      "(`in` = inside a `stablehlo.while` body, `out` = outside):\n")
    A("| row | size | config | cholesky in/out | triangular_solve in/out | syevd | custom_call in/out |")
    A("|---|---|---|---|---|---|---|")
    for r in f09:
        for cfg, ir in (("with std. opt.", r["ir_B"]), ("without", r["ir_A"])):
            if not ir:
                continue
            def g(n):
                d = ir.get(n, {})
                return "%s/%s" % (d.get("in", "-"), d.get("out", "-"))
            A("| %s | %s | %s | %s | %s | %s | %s |"
              % (r["model"], r["size"], cfg, g("stablehlo.cholesky"),
                 g("triangular_solve"),
                 ir.get("syevd", {}).get("total", "-"), g("custom_call")))
    A("\nPer-cell load records (sweep cells only):\n")
    for r in f09:
        if r["source"] != "sweep":
            continue
        A("- `%s %s` with std. opt.: %s" % (r["model"], r["size"], fmt_load(r["load_B"])))
        A("- `%s %s` without: %s" % (r["model"], r["size"], fmt_load(r["load_A"])))

    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", out, "(%d fig10 rows, %d fig09 rows)" % (len(f10), len(f09)))


if __name__ == "__main__":
    main()
