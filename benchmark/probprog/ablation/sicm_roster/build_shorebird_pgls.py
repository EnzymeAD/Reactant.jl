#!/usr/bin/env python3
"""Build the shorebird PGLS data set used by the TriSolveScale row.

Source: the `shorebird` data of the R package **caper** (Orme et al.), i.e.
`data/shorebird.rda` of github.com/davidorme/caper, which ships
`shorebird.tree` (an ape `phylo`, 71 tips) and `shorebird.data` (71 species x
Species, M.Mass, F.Mass, Egg.Mass, Cl.size, Mat.syst). The traits are from
Lislevand & Thomas (2006/2007), "Limited male incubation ability and the
evolution of egg size in shorebirds".

Model mirrored: the caper vignette's simple PGLS,
`pgls(log(Egg.Mass) ~ log(M.Mass), shorebird)` (vignettes/caper.rnw:530), i.e.
Brownian motion, lambda = 1, so the correlation structure is the plain
phylogenetic variance-covariance of the tree and is FIXED data.

The tree is converted from ape's (edge, edge.length) representation to Newick
and then run through the Newick parser of
`pplbench/pplbench/models/phylo_pois.py`, so both phylogenetic rows in this
benchmark build their covariance with identical code. The result is
cross-checked against a direct edge-matrix computation of the same quantity.

Writes pplbench/pplbench/models/real_data/shorebird_pgls.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp/sbrantq-rda")
import rdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(BENCH, "pplbench", "pplbench", "models")
sys.path.insert(0, os.path.join(BENCH, "pplbench"))
from pplbench.models.phylo_pois import _parse_newick  # noqa: E402

RDA = "/mnt4/sbrantq/probprog/caper/data/shorebird.rda"
OUT = os.path.join(MODELS, "real_data", "shorebird_pgls.json")


def to_newick(edge, edge_length, tip_label, nnode):
    """ape (edge, edge.length) -> Newick string. Tips are 1..Ntip."""
    ntip = len(tip_label)
    children = {}
    blen = {}
    parents = set()
    childset = set()
    for (p, c), L in zip(edge, edge_length):
        p, c = int(p), int(c)
        children.setdefault(p, []).append(c)
        blen[c] = float(L)
        parents.add(p)
        childset.add(c)
    roots = sorted(parents - childset)
    assert len(roots) == 1, roots
    root = roots[0]

    def emit(node):
        if node <= ntip:
            s = str(tip_label[node - 1])
        else:
            s = "(" + ",".join(emit(c) for c in children[node]) + ")"
        if node in blen:
            s += ":%.12g" % blen[node]
        return s

    return emit(root) + ";"


def vcv_from_newick(newick):
    """Same semantics as phylo_pois.vcv_from_nexus: C[i,j] = depth of the MRCA
    of tips i and j measured from the root, C[i,i] = root-to-tip distance."""
    root = _parse_newick(newick)
    tips = []

    def walk(nd, depth, path_ids, path_depths):
        d = depth + nd["len"]
        ids = path_ids + [id(nd)]
        ds = path_depths + [d]
        if not nd["children"]:
            tips.append((nd["name"], dict(zip(ids, ds))))
        for c in nd["children"]:
            walk(c, d, ids, ds)

    walk(root, 0.0, [], [])
    names = [t[0] for t in tips]
    n = len(tips)
    C = np.zeros((n, n))
    for i in range(n):
        ai = tips[i][1]
        for j in range(n):
            aj = tips[j][1]
            C[i, j] = max(ai[k] for k in ai if k in aj)
    return C, names


def vcv_direct(edge, edge_length, tip_label):
    """Independent check: root-to-node depths from the edge list, then
    C[i,j] = depth of the deepest common ancestor of tips i and j."""
    ntip = len(tip_label)
    children = {}
    blen = {}
    parents, childset = set(), set()
    for (p, c), L in zip(edge, edge_length):
        p, c = int(p), int(c)
        children.setdefault(p, []).append(c)
        blen[c] = float(L)
        parents.add(p)
        childset.add(c)
    root = sorted(parents - childset)[0]
    depth, anc = {root: 0.0}, {root: []}
    stack = [root]
    while stack:
        u = stack.pop()
        for c in children.get(u, []):
            depth[c] = depth[u] + blen[c]
            anc[c] = anc[u] + [u]
            stack.append(c)
    n = ntip
    C = np.zeros((n, n))
    for i in range(1, n + 1):
        pi = anc[i] + [i]
        si = set(pi)
        for j in range(1, n + 1):
            pj = anc[j] + [j]
            common = [k for k in pj if k in si]
            C[i - 1, j - 1] = max(depth[k] for k in common)
    return C


def main():
    conv = rdata.conversion.convert(rdata.parser.parse_file(RDA))
    tree = conv["shorebird.tree"]
    df = conv["shorebird.data"]

    edge = np.asarray(tree["edge"])
    elen = np.asarray(tree["edge.length"])
    tips = [str(s) for s in np.asarray(tree["tip.label"])]
    nnode = int(np.asarray(tree["Nnode"])[0])

    newick = to_newick(edge, elen, tips, nnode)
    C_nwk, names = vcv_from_newick(newick)
    C_dir = vcv_direct(edge, elen, tips)
    # vcv_direct is in tip order 1..Ntip; reorder it to the Newick walk order
    idx = [tips.index(nm) for nm in names]
    C_dir = C_dir[np.ix_(idx, idx)]
    agree = float(np.max(np.abs(C_nwk - C_dir)))
    print("newick-parser vcv vs direct edge-matrix vcv: max abs diff = %.3g" % agree)
    assert agree < 1e-9, "the two vcv constructions disagree"

    # Align the traits to the tree's tip order.
    species = [str(s) for s in df["Species"].values]
    pos = {s: i for i, s in enumerate(species)}
    missing = [nm for nm in names if nm not in pos]
    assert not missing, "tips absent from the trait table: %s" % missing[:5]
    rows = [pos[nm] for nm in names]
    egg = np.asarray(df["Egg.Mass"].values, dtype=np.float64)[rows]
    mmass = np.asarray(df["M.Mass"].values, dtype=np.float64)[rows]
    fmass = np.asarray(df["F.Mass"].values, dtype=np.float64)[rows]
    assert np.all(np.isfinite(egg)) and np.all(np.isfinite(mmass))

    y = np.log(egg)
    x = np.log(mmass)
    X = np.column_stack([np.ones(len(y)), x])
    n, k = X.shape

    # Ultrametric? (root-to-tip constant) and conditioning.
    d = np.diag(C_nwk)
    print("root-to-tip depth: min %.6g max %.6g  (ultrametric: %s)"
          % (d.min(), d.max(), bool(np.allclose(d, d[0]))))
    C = C_nwk / d.max()          # unit diagonal for an ultrametric tree
    w = np.linalg.eigvalsh(C)
    print("n=%d k=%d  cond(C)=%.4g  eig(C)=[%.4g, %.4g]  chol ok=%s"
          % (n, k, w.max() / w.min(), w.min(), w.max(),
             _chol_ok(C)))

    # GLS reference slope (what the fitted model should reproduce).
    L = np.linalg.cholesky(C)
    Xt = np.linalg.solve(L, X)
    yt = np.linalg.solve(L, y)
    bg = np.linalg.lstsq(Xt, yt, rcond=None)[0]
    bo = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = yt - Xt @ bg
    s2 = float(resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(Xt.T @ Xt)
    se = np.sqrt(s2 * np.diag(XtX_inv))
    print("GLS (PGLS, lambda=1) beta = [%.4f, %.4f]  se = [%.4f, %.4f]"
          % (bg[0], bg[1], se[0], se[1]))
    print("OLS                  beta = [%.4f, %.4f]" % (bo[0], bo[1]))

    prov = (
        "Shorebird phylogeny and traits from the R package caper "
        "(github.com/davidorme/caper, data/shorebird.rda): shorebird.tree, an "
        "ape phylo with %d tips and %d internal nodes, and shorebird.data, %d "
        "species x (Species, M.Mass, F.Mass, Egg.Mass, Cl.size, Mat.syst). "
        "Traits from Lislevand & Thomas, 'Limited male incubation ability and "
        "the evolution of egg size in shorebirds' (Biology Letters, 2006/2007). "
        "Model mirrored: the caper vignette's simple PGLS "
        "pgls(log(Egg.Mass) ~ log(M.Mass), shorebird) at the default lambda=1 "
        "(Brownian motion), so C is the plain phylogenetic variance-covariance "
        "of the tree and is fixed data. C[i,j] = root-to-MRCA path length, "
        "computed by converting ape's (edge, edge.length) to Newick and running "
        "it through the Newick parser of models/phylo_pois.py, cross-checked "
        "against a direct edge-matrix computation (max abs diff %.3g); then "
        "divided by the root-to-tip depth so diag(C) = 1. Nothing is simulated: "
        "y = log(Egg.Mass) and X = [1, log(M.Mass)] are the recorded traits. "
        "Built by benchmark/probprog/ablation/sicm_roster/build_shorebird_pgls.py."
        % (len(tips), nnode, len(species), agree)
    )
    out = {
        "_provenance": prov,
        "N": int(n),
        "K": int(k),
        "species": names,
        "y": y.tolist(),
        "X": X.tolist(),
        "C": C.tolist(),
        "log_F_Mass": np.log(fmass).tolist(),
        "gls_beta_reference": bg.tolist(),
        "gls_se_reference": se.tolist(),
        "ols_beta_reference": bo.tolist(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, "(%.2f MB)" % (os.path.getsize(OUT) / 1e6))


def _chol_ok(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    main()
