#!/usr/bin/env python3
"""Build the Columbus spatial-lag (SAR) data set: the n=49 fallback for the
DotScale row after the n=506 Boston instance proved impractical.

Source: `columbus` of the R package **spdep** (github.com/r-spatial/spdep,
data/columbus.rda): the 49 planning neighbourhoods of Columbus, Ohio, from
Anselin (1988), *Spatial Econometrics: Methods and Models*, with `col.gal.nb`,
the queen contiguity neighbours list distributed with them (also shipped as
inst/etc/weights/columbus.gal).

Model: the textbook Columbus spatial-lag regression,
CRIME ~ INC + HOVAL, i.e. residential burglaries and vehicle thefts per 1000
households against household income and housing value, which is the example
spdep's own `lagsarlm` documentation uses.

Weight matrices follow exactly the same construction and rationale as
build_boston_sar.py: W = D^-1 B is the row-standardised lag operator, and
Ws = D^-1/2 B D^-1/2 is the symmetric normalised adjacency used for the
log-determinant (similar to W, so identical spectrum and identical
determinant). See real_data/README.md.

Writes pplbench/pplbench/models/real_data/columbus_sar.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp/sbrantq-rda")
import rdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(BENCH, "pplbench", "pplbench", "models", "real_data",
                   "columbus_sar.json")
RDA = "/mnt/sbrantq/probprog/spdep/data/columbus.rda"


def main():
    conv = rdata.conversion.convert(rdata.parser.parse_file(RDA))
    df = conv["columbus"]
    nb = conv["col.gal.nb"]
    n = len(nb)
    assert n == df.shape[0] == 49, (n, df.shape)

    B = np.zeros((n, n))
    for i, row in enumerate(nb):
        for j in np.asarray(row).ravel():
            j = int(j) - 1
            if j >= 0 and j != i:
                B[i, j] = 1.0
    asym = float(np.max(np.abs(B - B.T)))
    B = np.maximum(B, B.T)
    deg = B.sum(1)
    edges = int(B.sum() // 2)
    assert deg.min() > 0, "isolated unit"

    W = B / deg[:, None]
    Dm = np.diag(1.0 / np.sqrt(deg))
    Ws = 0.5 * ((Dm @ B @ Dm) + (Dm @ B @ Dm).T)
    es = np.linalg.eigvalsh(Ws)
    er = np.sort(np.linalg.eigvals(W).real)
    spec_gap = float(np.max(np.abs(np.sort(es) - er)))
    lo, hi = 1.0 / float(es.min()), 1.0 / float(es.max())
    print("n=%d edges=%d degree min/med/max=%d/%d/%d asym_before_sym=%.3g"
          % (n, edges, deg.min(), np.median(deg), deg.max(), asym))
    print("eig(Ws) = [%.6f, %.6f]; same spectrum as row-standardised W "
          "(max abs diff %.3g); stationary rho in (%.4f, %.4f)"
          % (es.min(), es.max(), spec_gap, lo, hi))

    det_err = 0.0
    for rho in np.linspace(lo + 1e-3, hi - 1e-3, 40):
        a = np.linalg.slogdet(np.eye(n) - rho * W)[1]
        b = np.linalg.slogdet(np.eye(n) - rho * Ws)[1]
        det_err = max(det_err, abs(a - b))
    print("determinant identity max abs diff = %.3g" % det_err)
    assert det_err < 1e-8

    y = np.asarray(df["CRIME"].values, dtype=np.float64)
    feats = [("INC", np.asarray(df["INC"].values, dtype=np.float64)),
             ("HOVAL", np.asarray(df["HOVAL"].values, dtype=np.float64))]
    names = ["(Intercept)"] + [f[0] for f in feats]
    Z = np.column_stack([f[1] for f in feats])
    Z = (Z - Z.mean(0)) / Z.std(0)
    ym, ys = y.mean(), y.std()
    yz = (y - ym) / ys                     # standardise the response too: CRIME
    X = np.column_stack([np.ones(n), Z])   # is on a 0-70 scale, poor for NUTS
    k = X.shape[1]
    print("k=%d  cond(X'X)=%.4g  CRIME mean %.4f sd %.4f (response z-scored)"
          % (k, np.linalg.cond(X.T @ X), ym, ys))

    for rho in (-1.0, -0.5, 0.0, 0.5, 0.9, 0.99):
        A = np.eye(n) - rho * Ws
        ev = np.linalg.eigvalsh(A)
        ok = True
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            ok = False
        print("   rho=%+.2f eig(I - rho Ws) in [%.4g, %.4g] cond=%.4g chol ok=%s"
              % (rho, ev.min(), ev.max(), ev.max() / ev.min(), ok))

    def concentrated_ll(rho):
        A = np.eye(n) - rho * W
        Ay = A @ yz
        b = np.linalg.lstsq(X, Ay, rcond=None)[0]
        r = Ay - X @ b
        s2 = float(r @ r) / n
        L = np.linalg.cholesky(np.eye(n) - rho * Ws)
        logdet = 2.0 * float(np.log(np.diag(L)).sum())
        return logdet - 0.5 * n * np.log(2 * np.pi * s2) - 0.5 * n, b, s2

    grid = np.linspace(lo + 1e-3, 0.999, 400)
    vals = [concentrated_ll(r)[0] for r in grid]
    rho_hat = float(grid[int(np.argmax(vals))])
    _, b_hat, s2_hat = concentrated_ll(rho_hat)
    print("profile-likelihood rho_hat = %.4f, sigma_hat = %.4f"
          % (rho_hat, np.sqrt(s2_hat)))
    for nm, bv in zip(names, b_hat):
        print("     beta[%-12s] = %+.4f" % (nm, bv))

    prov = (
        "Columbus, Ohio crime data, %d planning neighbourhoods. Source: the "
        "`columbus` data of the R package spdep (github.com/r-spatial/spdep, "
        "data/columbus.rda), from Anselin (1988), Spatial Econometrics: "
        "Methods and Models, with `col.gal.nb`, the queen contiguity "
        "neighbours list distributed with them (%d undirected edges, degrees "
        "%d-%d). Model: the textbook spatial-lag regression CRIME ~ INC + "
        "HOVAL used by spdep's own lagsarlm documentation. The response CRIME "
        "and both regressors are z-scored (a sampler-conditioning choice that "
        "rescales beta and sigma; it does not change the model). Two fixed "
        "weight matrices, both data: W = D^-1 B row-standardised, the lag "
        "operator; Ws = D^-1/2 B D^-1/2 symmetric normalised adjacency, "
        "similar to W and so of identical spectrum, used for the "
        "log-determinant so it can be a Cholesky of a symmetric positive "
        "definite matrix (Ord 1975; spdep's `similar.listw`); identity checked "
        "to %.3g. eig(Ws) = [%.6f, %.6f], stationary rho in (%.4f, 1). This "
        "n=49 instance is the disclosed fallback for the DotScale row after "
        "the n=506 Boston instance proved impractical to measure (its "
        "SICM-disabled arm exceeded 40 min per trial). Nothing is simulated. "
        "Built by benchmark/probprog/ablation/sicm_roster/build_columbus_sar.py."
        % (n, edges, int(deg.min()), int(deg.max()), det_err, es.min(),
           es.max(), lo)
    )
    out = {
        "_provenance": prov,
        "N": int(n), "K": int(k),
        "feature_names": names,
        "y": yz.tolist(),
        "X": X.tolist(),
        "W": W.tolist(),
        "Ws": Ws.tolist(),
        "rho_lo": float(lo), "rho_hi": 1.0,
        "y_mean": float(ym), "y_sd": float(ys),
        "profile_rho_hat": rho_hat,
        "profile_sigma_hat": float(np.sqrt(s2_hat)),
        "profile_beta_hat": b_hat.tolist(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, "(%.2f MB)" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
