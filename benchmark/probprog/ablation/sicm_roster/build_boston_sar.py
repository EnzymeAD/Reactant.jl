#!/usr/bin/env python3
"""Build the Boston spatial-lag (SAR) data set used by the DotScale row.

Source: `boston` of the R package **spData** (github.com/Nowosad/spData,
data/boston.rda), which carries `boston.c`, the 506-tract Harrison & Rubinfeld
(1978) Boston housing table with Gilley & Pace's corrected median value CMEDV,
and `boston.soi`, the sphere-of-influence neighbours list built for these
tracts (an spdep `nb` object, symmetric by construction).

Model: the Bayesian spatial-lag (SAR) model of Anselin (1988) / LeSage & Pace
(2009),  (I - rho W) y = X beta + eps,  eps ~ N(0, sigma^2 I),
with W fixed data and rho, beta, sigma sampled.

Regressors: the Harrison & Rubinfeld hedonic specification as used in spdep's
own Boston examples, y = log(CMEDV) on
CRIM, ZN, INDUS, CHAS, NOX^2, RM^2, AGE, log(DIS), log(RAD), TAX, PTRATIO, B,
log(LSTAT). Non-intercept columns are z-scored; that is a conditioning
convenience for the sampler and rescales beta, it does not change the model.

Two fixed weight matrices are shipped, and both are data:

  W  = D^-1 B, the ROW-STANDARDISED sphere-of-influence contiguity. This is the
       lag operator of the model equation and the convention the Boston SAR
       literature reports rho in.
  Ws = D^-1/2 B D^-1/2, the SYMMETRIC normalised adjacency. It is similar to W
       (Ws = D^1/2 W D^-1/2), hence has exactly the same spectrum, hence
       log|I - rho W| = log|I - rho Ws| for every rho.

The model's log-determinant term is therefore computed as a Cholesky of
I - rho Ws, which is symmetric positive definite on the stationary interval,
while the lag term uses W. This is not a modelling shortcut: it is the standard
computational device for the SAR determinant (Ord 1975; spdep's
`similar.listw`), used precisely because the symmetric form makes the
determinant tractable. The identity was verified numerically to 2.8e-14 over
rho in [-1, 0.99] when this file was built.

Writes pplbench/pplbench/models/real_data/boston_sar.json.
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
                   "boston_sar.json")
RDA = "/mnt/sbrantq/probprog/spData/data/boston.rda"


def main():
    conv = rdata.conversion.convert(rdata.parser.parse_file(RDA))
    bc = conv["boston.c"]
    soi = conv["boston.soi"]
    n = len(soi)
    assert n == bc.shape[0] == 506, (n, bc.shape)

    # Binary symmetric contiguity from the 1-based neighbour list.
    B = np.zeros((n, n))
    for i, nb in enumerate(soi):
        for j in np.asarray(nb).ravel():
            j = int(j) - 1
            if j >= 0 and j != i:
                B[i, j] = 1.0
    asym = float(np.max(np.abs(B - B.T)))
    B = np.maximum(B, B.T)          # no-op if already symmetric
    deg = B.sum(1)
    edges = int(B.sum() // 2)
    assert deg.min() > 0, "isolated tract"

    W = B / deg[:, None]                                   # row-standardised
    Dm = np.diag(1.0 / np.sqrt(deg))
    Ws = Dm @ B @ Dm                                       # symmetric, similar to W
    Ws = 0.5 * (Ws + Ws.T)
    es = np.linalg.eigvalsh(Ws)
    er = np.sort(np.linalg.eigvals(W).real)
    spec_gap = float(np.max(np.abs(np.sort(es) - er)))
    lo, hi = 1.0 / float(es.min()), 1.0 / float(es.max())
    print("n=%d edges=%d degree min/med/max=%d/%d/%d asym_before_sym=%.3g"
          % (n, edges, deg.min(), np.median(deg), deg.max(), asym))
    print("eig(Ws) = [%.6f, %.6f]; same spectrum as row-standardised W "
          "(max abs diff %.3g); stationary rho in (%.4f, %.4f)"
          % (es.min(), es.max(), spec_gap, lo, hi))

    # log|I - rho W| == log|I - rho Ws| (similar matrices), which is what lets
    # the model use a Cholesky for the determinant.
    det_err = 0.0
    for rho in np.linspace(lo + 1e-3, hi - 1e-3, 40):
        a = np.linalg.slogdet(np.eye(n) - rho * W)[1]
        b = np.linalg.slogdet(np.eye(n) - rho * Ws)[1]
        det_err = max(det_err, abs(a - b))
    print("determinant identity |log|I-rho W| - log|I-rho Ws|| max = %.3g" % det_err)
    assert det_err < 1e-8

    col = {c: np.asarray(bc[c].values, dtype=np.float64) for c in
           ["CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]}
    y = np.log(col["CMEDV"])
    feats = [
        ("CRIM", col["CRIM"]),
        ("ZN", col["ZN"]),
        ("INDUS", col["INDUS"]),
        ("CHAS", col["CHAS"]),
        ("NOX_sq", col["NOX"] ** 2),
        ("RM_sq", col["RM"] ** 2),
        ("AGE", col["AGE"]),
        ("log_DIS", np.log(col["DIS"])),
        ("log_RAD", np.log(col["RAD"])),
        ("TAX", col["TAX"]),
        ("PTRATIO", col["PTRATIO"]),
        ("B", col["B"]),
        ("log_LSTAT", np.log(col["LSTAT"])),
    ]
    names = ["(Intercept)"] + [f[0] for f in feats]
    Z = np.column_stack([f[1] for f in feats])
    Z = (Z - Z.mean(0)) / Z.std(0)
    X = np.column_stack([np.ones(n), Z])
    k = X.shape[1]
    print("k=%d  cond(X'X)=%.4g  y mean %.4f sd %.4f"
          % (k, np.linalg.cond(X.T @ X), y.mean(), y.std()))

    # SPD check for I - rho Ws across the interval the sampler can reach.
    for rho in (-1.0, -0.5, 0.0, 0.5, 0.8, 0.95, 0.99):
        A = np.eye(n) - rho * Ws
        ev = np.linalg.eigvalsh(A)
        ok = True
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            ok = False
        print("   rho=%+.2f  eig(I - rho Ws) in [%.4g, %.4g]  cond=%.4g  chol ok=%s"
              % (rho, ev.min(), ev.max(), ev.max() / ev.min(), ok))

    # Reference fit: profile the SAR likelihood over rho on a grid (the
    # qualitative result the posterior should reproduce). Lag from W,
    # determinant from Ws, exactly as the model computes it.
    def concentrated_ll(rho):
        A = np.eye(n) - rho * W
        Ay = A @ y
        b = np.linalg.lstsq(X, Ay, rcond=None)[0]
        r = Ay - X @ b
        s2 = float(r @ r) / n
        L = np.linalg.cholesky(np.eye(n) - rho * Ws)
        logdet = 2.0 * float(np.log(np.diag(L)).sum())
        return logdet - 0.5 * n * np.log(2 * np.pi * s2) - 0.5 * n, b, s2

    grid = np.linspace(lo + 1e-3, 0.999, 400)
    vals = [concentrated_ll(r)[0] for r in grid]
    rho_hat = float(grid[int(np.argmax(vals))])
    ll, b_hat, s2_hat = concentrated_ll(rho_hat)
    print("profile-likelihood rho_hat = %.4f, sigma_hat = %.4f" % (rho_hat, np.sqrt(s2_hat)))
    for nm, bv in zip(names, b_hat):
        print("     beta[%-12s] = %+.4f" % (nm, bv))

    prov = (
        "Boston housing spatial-lag data. Source: the `boston` data of the R "
        "package spData (github.com/Nowosad/spData, data/boston.rda): "
        "`boston.c`, the 506-tract Harrison & Rubinfeld (1978) table with "
        "Gilley & Pace's corrected median value CMEDV, and `boston.soi`, the "
        "sphere-of-influence neighbours list for those tracts (spdep `nb`, "
        "symmetric, %d undirected edges, degrees %d-%d). Response y = "
        "log(CMEDV); regressors are the Harrison & Rubinfeld hedonic set used "
        "in spdep's Boston examples (CRIM, ZN, INDUS, CHAS, NOX^2, RM^2, AGE, "
        "log(DIS), log(RAD), TAX, PTRATIO, B, log(LSTAT)), z-scored, with an "
        "intercept prepended. Two fixed weight matrices, both data: W = D^-1 B "
        "is the row-standardised contiguity and is the lag operator of the "
        "model equation (the convention the Boston SAR literature reports rho "
        "in); Ws = D^-1/2 B D^-1/2 is the symmetric normalised adjacency, "
        "similar to W and therefore of identical spectrum, used for the "
        "log-determinant so that it can be taken as a Cholesky of a symmetric "
        "positive definite matrix (Ord 1975; spdep's `similar.listw`). The "
        "identity log|I - rho W| = log|I - rho Ws| was checked over the "
        "stationary interval to %.3g. eig(Ws) = [%.6f, %.6f], so the "
        "stationary interval is rho in (%.4f, 1). Nothing is simulated: y and "
        "X are the recorded values. Built by "
        "benchmark/probprog/ablation/sicm_roster/build_boston_sar.py."
        % (edges, int(deg.min()), int(deg.max()), det_err, es.min(), es.max(), lo)
    )
    out = {
        "_provenance": prov,
        "N": int(n),
        "K": int(k),
        "feature_names": names,
        "y": y.tolist(),
        "X": X.tolist(),
        "W": W.tolist(),
        "Ws": Ws.tolist(),
        "rho_lo": float(lo),
        "rho_hi": 1.0,
        "profile_rho_hat": rho_hat,
        "profile_sigma_hat": float(np.sqrt(s2_hat)),
        "profile_beta_hat": b_hat.tolist(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, "(%.2f MB)" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
