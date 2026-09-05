#!/usr/bin/env python3
"""Emit the markdown tables of the roster results file from the aggregate JSON.

Usage: report_tables.py <raw.json> [headline_batch]
"""
import json
import sys

ROWS = [
    ("car_spatial_scotland", "CholScale", "Leroux CAR, Scotland lip cancer", "n=56"),
    ("car_spatial_nyc", "CholScale", "Leroux CAR, NYC traffic sub-map", "n=250"),
    ("hierarchical_mvn", "OuterProdScale", "Hierarchical MVN", "K=60, J=2"),
    ("lmm_radon", "EigenLift", "Marginalized LMM, MN radon", "n=250"),
    ("phylo_pois", "CholScale", "Phylogenetic Poisson regression", "n=200"),
]


def main():
    raw = json.load(open(sys.argv[1]))
    head = sys.argv[2] if len(sys.argv) > 2 else None
    summary = raw["summary"]
    batches = sorted(raw["batches"])

    print("### Headline table (batch %s, last-trial statistic)\n" % head)
    print("| Pattern | Model | Size | SICM (s) | No SICM (s) | Speedup |")
    print("|---|---|---|---|---|---|")
    for cell, pat, label, size in ROWS:
        r = summary.get(cell, {}).get(head)
        if not r:
            continue
        print("| %s | %s | %s | %.2f | %.2f | %.1fx |"
              % (pat, label, size, r["sicm_last_s"], r["nosicm_last_s"],
                 r["speedup_last"]))

    print("\n### Both statistics, every batch\n")
    print("| Model | Batch | SICM trials (s) | SICM last | SICM median | "
          "No SICM trials (s) | No SICM last | No SICM median | "
          "Speedup last | Speedup median |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cell, pat, label, size in ROWS:
        for tag in batches:
            r = summary.get(cell, {}).get(tag)
            if not r:
                continue
            print("| %s | %s | %s | %.3f | %.3f | %s | %.3f | %.3f | %.1fx | %.1fx |"
                  % (label, tag,
                     "; ".join("%.3f" % t for t in r["sicm_trials"]),
                     r["sicm_last_s"], r["sicm_median_s"],
                     "; ".join("%.3f" % t for t in r["nosicm_trials"]),
                     r["nosicm_last_s"], r["nosicm_median_s"],
                     r["speedup_last"], r["speedup_median"]))

    print("\n### Mixing / movement gate (post-warmup, both arms, every batch)\n")
    print("| Model | Batch | Arm | n_eff min/med/max | Rhat min/med/max | "
          "min unique draws per chain (of N) | PLL[-1] per chain |")
    print("|---|---|---|---|---|---|---|")
    for cell, pat, label, size in ROWS:
        for tag in batches:
            c = raw["batches"][tag]["cells"].get(cell)
            if not c or "arms" not in c:
                continue
            for arm, a in c["arms"].items():
                mv = a["movement"]
                worst = min((min(m["unique_per_chain"]), v)
                            for v, m in mv.items())
                ndraw = mv[worst[1]]["n_draws"]
                print("| %s | %s | %s | %.1f / %.1f / %.1f | %.4f / %.4f / %.4f | "
                      "%d (%s, of %d) | %s |"
                      % (label, tag, arm, a["neff"][0], a["neff"][1], a["neff"][2],
                         a["rhat"][0], a["rhat"][1], a["rhat"][2],
                         worst[0], worst[1], ndraw,
                         ", ".join("%.2f" % x for x in a["pll_last_per_chain"])))

    print("\n### Machine record\n")
    print("| Batch | md5 start | md5 end | load1 at start | load1 at end | window |")
    print("|---|---|---|---|---|---|")
    for tag in batches:
        m = raw["batches"][tag]["meta"]
        print("| %s | %s | %s | %s | %s | %s -> %s |"
              % (tag, m.get("md5 at batch start", "?")[:8],
                 m.get("md5 at batch end", "?")[:8],
                 m.get("loadavg at batch start", "?").split()[0],
                 m.get("loadavg at batch end", "?").split()[0],
                 m.get("start", "?"), m.get("end", "?")))

    print("\n### Per-cell load record\n")
    print("| Batch | Cell | load1 before | load1 after | window | script exit |")
    print("|---|---|---|---|---|---|")
    for tag in batches:
        for cell, pat, label, size in ROWS:
            c = raw["batches"][tag]["cells"].get(cell)
            if not c:
                continue
            bm = c.get("batch_cell_meta", {})
            if not bm:
                continue
            print("| %s | %s | %s | %s | %s -> %s | %s |"
                  % (tag, cell, bm.get("loadavg before", "?").split()[0],
                     bm.get("loadavg after", "?").split()[0],
                     bm.get("start", "?"), bm.get("end", "?"),
                     bm.get("script exit", "?")))


if __name__ == "__main__":
    main()
