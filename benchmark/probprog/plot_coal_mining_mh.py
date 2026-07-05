#!/usr/bin/env python3
"""Plot the branchy change-point posterior p(s|y) vs the true switch s*.

Reads benchmark/probprog/figures/coal_mining_mh_posterior.json (written by
coal_mining_mh_demo.jl) and renders coal_mining_mh_posterior.png. p(s|y) is
recovered by enumerating s through the verified branchy `generate` path (the
intended MH/regenerate kernel is blocked by a C2 bug; see the demo header).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def main():
    with open(os.path.join(FIG_DIR, "coal_mining_mh_posterior.json")) as f:
        d = json.load(f)

    s = d["s"]
    p = d["p"]
    s_true = d["s_true"]
    map_s = d["map_s"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(s, p, width=0.9, color="#c44", label="p(s | y)")
    ax.axvline(
        s_true, color="k", linestyle="--", linewidth=1.5,
        label=f"true s* = {s_true}",
    )
    ax.set_xlabel("switch position s")
    ax.set_ylabel("posterior probability")
    ax.set_title(
        f"Coal-mining change point: branchy observe-inside-if model "
        f"(MAP s = {map_s})"
    )
    ax.set_xlim(0, d["n"])
    ax.legend()
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "coal_mining_mh_posterior.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

    lo, hi = max(0, s_true - 6), min(d["n"], s_true + 6)
    mass = sum(p[lo : hi + 1])
    print(f"posterior mass in [{lo},{hi}] around s*={s_true}: {mass:.4f}")


if __name__ == "__main__":
    main()
