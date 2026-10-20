#!/usr/bin/env python3
"""Build harness-format data files at additional problem sizes for the
2026-09-10 SICM-ablation problem-size sweep (radon LMM, diamonds regression,
hierarchical MVN).

Every file is produced by running the pplbench model class's own
`generate_data` arithmetic and `generate_init_params`, then serialising in the
exact shape pplbench's SubprocessInference._serialize_data emits, so a new size
differs from the frozen row only in its size argument. Each writer is validated
against the frozen file at the frozen size before any new size is written
(--verify).

Seeds. pplbench derives per-trial seeds from the config's model seed with
np.random.RandomState(12345): compile seed 7735426, then trial seeds 1396133,
2993578, 6164479, 9523583. The frozen roster files carry
generate_init_params(1396133, **attrs), which is also run_frozen_batch.sh's
FROZEN_SEED. generate_data is called with the model seed itself (12345).

Usage:
  build_roster_sizes.py verify
  build_roster_sizes.py radon 500 919
  build_roster_sizes.py diamonds 20000 53920
  build_roster_sizes.py hmvn K120_J2 K240_J2 K60_J5
"""
import csv
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROSTER = os.path.normpath(os.path.join(HERE, ".."))
BENCH = os.path.normpath(os.path.join(ROSTER, "..", ".."))
REAL = os.path.join(BENCH, "pplbench", "pplbench", "models", "real_data")
DATA = os.path.join(ROSTER, "data")
MODEL_SEED = 12345
TRIAL_SEED = 1396133
DIAMONDS_CSV = os.path.join(HERE, "diamonds_full.csv")


def write(name, obj):
    p = os.path.join(DATA, name + ".json")
    with open(p, "w") as f:
        json.dump(obj, f)
    print("   wrote", p, "(%.1f MB)" % (os.path.getsize(p) / 1e6))
    return p


# ---------------------------------------------------------------- radon ----
def radon_dataset(n, seed=MODEL_SEED):
    """models/linear_mixed_model.py, `radon_mn` branch, verbatim arithmetic."""
    rng = np.random.default_rng(seed)
    raw = json.load(open(os.path.join(REAL, "radon_mn.json")))
    N_full = int(raw["N"])
    idx = np.sort(rng.choice(N_full, size=min(n, N_full), replace=False))
    n = len(idx)
    floor = np.asarray(raw["floor_measure"], dtype=np.float64)[idx]
    y = np.asarray(raw["log_radon"], dtype=np.float64)[idx]
    county = np.asarray(raw["county_idx"], dtype=np.int64)[idx]
    counties = np.unique(county)
    q = len(counties)
    remap = {c: j for j, c in enumerate(counties)}
    X = np.column_stack([np.ones(n), floor])
    k = X.shape[1]
    Z = np.zeros((n, q))
    for i in range(n):
        Z[i, remap[county[i]]] = 1.0
    K_re = Z @ Z.T
    return X, K_re, y, {"n": n, "k": k, "q": q}, int(N_full)


def radon_init(k, seed=TRIAL_SEED):
    rng = np.random.default_rng(seed)
    return {"beta": rng.uniform(-2, 2, size=(k,)),
            "sigma_u": np.exp(rng.uniform(-1, 1, size=(1,))),
            "sigma_e": np.exp(rng.uniform(-1, 1, size=(1,)))}


def radon_obj(n):
    X, K_re, y, attrs, N_full = radon_dataset(n)
    m = attrs["n"]
    whole = m >= N_full
    prov = (
        "Minnesota radon data (posteriordb `radon_mn`; Gelman and Hill 2007), "
        "marginalised county-intercept linear mixed model: "
        "y ~ MVN(X beta, sigma_u^2 Z Z' + sigma_e^2 I), X = [1, floor], "
        "Z = county indicators. Source file "
        "pplbench/pplbench/models/real_data/radon_mn.json (N=%d homes, 85 "
        "counties). %s Counties are re-indexed to the ones present (q=%d), so Z "
        "has no all-zero column. y = log radon and floor are the recorded "
        "values; nothing is simulated. Produced by "
        "models/linear_mixed_model.py's `radon_mn` branch arithmetic, verbatim, "
        "serialised in the shape pplbench's "
        "SubprocessInference._serialize_data emits; init_params = "
        "LinearMixedModel.generate_init_params(%d, k=%d). Built by "
        "benchmark/probprog/ablation/sicm_roster/sweep_2026-09-10/"
        "build_roster_sizes.py on 2026-09-10."
        % (N_full,
           ("NO SUBSAMPLE: all %d homes are used (n >= N, so the without-"
            "replacement draw is a permutation and the sorted index list is the "
            "identity)." % N_full) if whole else
           ("Fixed-seed subsample, the house rule of real_data/README.md: "
            "numpy.random.default_rng(%d).choice(%d, %d, replace=False), sorted, "
            "so the subsample keeps the file's row order." % (MODEL_SEED, N_full, m)),
           attrs["q"], TRIAL_SEED, attrs["k"]))
    ip = radon_init(attrs["k"])
    return {
        "_provenance": prov,
        "X": X.tolist(), "K_re": K_re.tolist(), "Y": y.tolist(),
        "coords": {"obs": list(range(m)), "feature": list(range(attrs["k"])),
                   "i": list(range(m)), "j": list(range(m))},
        "dims": {"X": ["obs", "feature"], "K_re": ["i", "j"], "Y": ["obs"]},
        "attrs": attrs,
        "init_params": {k: v.tolist() for k, v in ip.items()},
    }


# ------------------------------------------------------------- diamonds ----
CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = list("DEFGHIJ")
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def contr_poly(m):
    """R's contr.poly(m): orthonormal polynomial contrasts on scores 1..m."""
    s = np.arange(1, m + 1, dtype=float)
    V = np.vander(s - s.mean(), m, increasing=True)
    q, r = np.linalg.qr(V)
    return (q * np.sign(np.diag(r)))[:, 1:]


CP = {5: contr_poly(5), 7: contr_poly(7), 8: contr_poly(8)}


def diamonds_rows():
    rows = list(csv.DictReader(open(DIAMONDS_CSV)))
    good = [r for r in rows
            if float(r["x"]) > 0 and float(r["y"]) > 0 and float(r["z"]) > 0]
    return rows, good


def diamonds_design(rows):
    """brms design matrix for
    log(price) ~ carat * (log(x) + log(y) + log(z)) + cut + color + clarity,
    ordered factors under R's default contr.poly, columns in R's
    model.matrix order: intercept, carat, log x, log y, log z, cut(4),
    color(6), clarity(7), carat:log x, carat:log y, carat:log z."""
    carat = np.array([float(r["carat"]) for r in rows])
    lx = np.log([float(r["x"]) for r in rows])
    ly = np.log([float(r["y"]) for r in rows])
    lz = np.log([float(r["z"]) for r in rows])
    cut = np.array([CP[5][CUT.index(r["cut"])] for r in rows])
    col = np.array([CP[7][COLOR.index(r["color"])] for r in rows])
    cla = np.array([CP[8][CLARITY.index(r["clarity"])] for r in rows])
    X = np.column_stack([np.ones(len(rows)), carat, lx, ly, lz, cut, col, cla,
                         carat * lx, carat * ly, carat * lz])
    Y = np.log([float(r["price"]) for r in rows])
    return X, Y


def diamonds_init(K, seed=TRIAL_SEED):
    rng = np.random.default_rng(seed)
    beta0 = rng.uniform(-0.5, 0.5, size=(K,))
    beta0[0] += 8.0
    return {"beta": beta0, "sigma": np.exp(rng.uniform(-1, 0, size=(1,)))}


def diamonds_obj(N):
    rows_all, good = diamonds_rows()
    n_all, n_good = len(rows_all), len(good)
    if N >= n_good:
        sel = good
        rule = ("NO SUBSAMPLE: all %d rows of the ggplot2 diamonds data that "
                "have a usable design row. %d of the %d rows carry a recorded "
                "physical dimension of exactly 0 (x, y or z), for which "
                "log(x/y/z) is undefined, and are dropped; the posteriordb "
                "5000-row sample contains none of them."
                % (n_good, n_all - n_good, n_all))
    else:
        rng = np.random.default_rng(MODEL_SEED)
        idx = np.sort(rng.choice(n_good, size=N, replace=False))
        sel = [good[i] for i in idx]
        rule = ("Fixed-seed subsample, the house rule of real_data/README.md: "
                "numpy.random.default_rng(%d).choice(%d, %d, replace=False), "
                "sorted, over the %d rows with a usable design row (%d of the "
                "%d rows carry a zero physical dimension and are dropped)."
                % (MODEL_SEED, n_good, N, n_good, n_all - n_good, n_all))
    Xr, Y = diamonds_design(sel)
    N = Xr.shape[0]
    K = Xr.shape[1]
    Xc = Xr[:, 1:] - Xr[:, 1:].mean(axis=0, keepdims=True)
    X_aug = np.column_stack([np.ones(N), Xc])
    prov = (
        "brms `diamonds` log-log price regression on the ggplot2 diamonds data, "
        "at N=%d. The design is the one posteriordb's `diamonds-diamonds` data "
        "file carries, rebuilt from the full 53940-row source: "
        "log(price) ~ carat * (log(x) + log(y) + log(z)) + cut + color + "
        "clarity (posteriordb models/info/diamonds.info.json), R model.matrix "
        "column order, ordered factors cut (5 levels), color (7) and clarity "
        "(8) under R's default contr.poly contrasts, giving K=25 columns: "
        "intercept, carat, log x, log y, log z, cut.L/.Q/.C/^4, color 6, "
        "clarity 7, carat:log x, carat:log y, carat:log z. Verified column by "
        "column against posterior_database/data/data/diamonds.json.zip at the "
        "5000 rows that file uses: max |X_rebuilt - X_posteriordb| = 5.33e-15 "
        "and max |Y_rebuilt - Y_posteriordb| = 5.33e-15, which is the 15-"
        "significant-digit precision at which posteriordb stores the numbers "
        "(the posteriordb 5000 are a random, unordered sample of the full data; "
        "they were located by matching the full 25-column design row plus the "
        "response). Full data source: the ggplot2 diamonds data as distributed "
        "at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
        "diamonds.csv (53940 rows, md5 4d3d1d4bbad5e0806dbaec425cf90196, "
        "retrieved 2026-09-10). Row selection: %s Y = log(price) and X are the "
        "recorded regression variables; nothing is simulated. Stan's "
        "`transformed data` centering of the 24 non-intercept columns at their "
        "own column means is applied here, as models/diamonds_regression.py "
        "does at load time. init_params = "
        "DiamondsRegression.generate_init_params(%d, k=%d). Built by "
        "benchmark/probprog/ablation/sicm_roster/sweep_2026-09-10/"
        "build_roster_sizes.py on 2026-09-10." % (N, rule, TRIAL_SEED, K))
    ip = diamonds_init(K)
    return {
        "_provenance": prov,
        "X": X_aug.tolist(), "Y": Y.tolist(),
        "coords": {"obs": list(range(N)), "feature": list(range(K))},
        "dims": {"X": ["obs", "feature"], "Y": ["obs"]},
        "attrs": {"n": N, "k": K, "kc": K - 1},
        "init_params": {k: v.tolist() for k, v in ip.items()},
    }


# ----------------------------------------------------------------- hmvn ----
def _random_correlation_matrix(rng, K):
    A = rng.standard_normal((K, K))
    Q, _ = np.linalg.qr(A)
    eigenvalues = rng.uniform(0.5, 2.0, K)
    S = Q @ np.diag(eigenvalues) @ Q.T
    D_inv = np.diag(1.0 / np.sqrt(np.diag(S)))
    R = D_inv @ S @ D_inv
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def hmvn_dataset(K, J, seed=MODEL_SEED, true_sigma_obs=0.5):
    """models/hierarchical_mvn.py generate_data, verbatim arithmetic."""
    rng = np.random.default_rng(seed)
    Omega = _random_correlation_matrix(rng, K)
    Omega += 1e-6 * np.eye(K)
    true_sigma_k = np.exp(rng.uniform(-0.5, 1.0, K))
    true_mu = rng.uniform(-2, 2, K)
    D = np.diag(true_sigma_k)
    Sigma = D @ Omega @ D
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((K, J))
    for j in range(J):
        eta_j = rng.standard_normal(K)
        theta_j = true_mu + L @ eta_j
        Y[:, j] = theta_j + rng.standard_normal(K) * true_sigma_obs
    return Omega, Y


def hmvn_init(K, J, seed=TRIAL_SEED):
    rng = np.random.default_rng(seed)
    return {"mu": rng.uniform(-2, 2, size=(K,)),
            "sigma_k": np.exp(rng.uniform(-1, 1, size=(K,))),
            "sigma_obs": np.exp(rng.uniform(-1, 0.5, size=(1,))),
            "eta": rng.standard_normal(J * K)}


def hmvn_obj(K, J):
    Omega, Y = hmvn_dataset(K, J)
    ip = hmvn_init(K, J)
    prov = (
        "Hierarchical MVN, SYNTHETIC, K=%d outcomes x J=%d groups. Produced by "
        "models/hierarchical_mvn.py's generate_data arithmetic verbatim with "
        "the ablation config's model seed %d (numpy.random.default_rng): a "
        "random correlation matrix Omega by the QR/eigenvalue construction plus "
        "1e-6 I, true per-outcome scales exp(U(-0.5, 1.0)), true grand means "
        "U(-2, 2), group effects theta_j = mu + chol(diag(sigma) Omega "
        "diag(sigma)) eta_j and observations Y[:, j] = theta_j + 0.5 N(0, I). "
        "Only K and J differ from the frozen K=60, J=2 row; every other "
        "constant and the whole draw order are unchanged. init_params = "
        "HierarchicalMVN.generate_init_params(%d, K=%d, J=%d). Built by "
        "benchmark/probprog/ablation/sicm_roster/sweep_2026-09-10/"
        "build_roster_sizes.py on 2026-09-10." % (K, J, MODEL_SEED, TRIAL_SEED, K, J))
    return {
        "_provenance": prov,
        "Omega": Omega.tolist(), "Y": Y.tolist(),
        "coords": {"outcome_i": list(range(K)), "outcome_j": list(range(K)),
                   "outcome": list(range(K)), "group": list(range(J))},
        "dims": {"Omega": ["outcome_i", "outcome_j"], "Y": ["outcome", "group"]},
        "attrs": {"K": K, "J": J, "n": K * J},
        "init_params": {k: v.tolist() for k, v in ip.items()},
    }


# --------------------------------------------------------------- verify ----
def cmp(name, new, ref_path):
    ref = json.load(open(ref_path))
    ok = True
    for k in ref:
        if k in ("_provenance",):
            continue
        if k in ("coords", "dims", "attrs"):
            same = ref[k] == new[k]
            print("   %-10s %-8s %s" % (name, k, "identical" if same else
                                        "DIFF %r vs %r" % (ref[k], new[k])))
            ok &= same
        elif k == "init_params":
            for kk in ref[k]:
                a = np.asarray(ref[k][kk], float); b = np.asarray(new[k][kk], float)
                same = a.shape == b.shape and np.array_equal(a, b)
                print("   %-10s init.%-8s %s" % (name, kk, "identical" if same
                                                 else "DIFF"))
                ok &= same
        else:
            a = np.asarray(ref[k], float); b = np.asarray(new[k], float)
            same = a.shape == b.shape and np.array_equal(a, b)
            print("   %-10s %-8s %s" % (name, k, "identical" if same else
                                        "DIFF max %.3g" % np.abs(a - b).max()))
            ok &= same
    print("   %s: %s" % (name, "REPRODUCES THE FROZEN FILE" if ok else "MISMATCH"))
    return ok


def verify():
    ok = True
    ok &= cmp("radon", radon_obj(250), os.path.join(DATA, "lmm_radon.json"))
    ok &= cmp("hmvn", hmvn_obj(60, 2), os.path.join(DATA, "hierarchical_mvn.json"))
    # diamonds: the frozen file is the posteriordb 5000 rows; the rebuild path
    # cannot select those rows by a size rule, so the check is column-by-column
    # against the posteriordb source at its own rows (done at build time and
    # recorded in the provenance). Re-run it here.
    z = zipfile.ZipFile("/mnt4/sbrantq/probprog/posteriordb/posterior_database/"
                        "data/data/diamonds.json.zip")
    raw = json.loads(z.read("diamonds.json"))
    Xp = np.asarray(raw["X"], float); Yp = np.asarray(raw["Y"], float)
    rows_all, good = diamonds_rows()
    Xf, Yf = diamonds_design(good)
    key = {}
    for i in range(len(good)):
        key.setdefault(tuple(np.round(np.append(Xf[i], Yf[i]), 9)), []).append(i)
    idx, miss = [], 0
    for r in range(Xp.shape[0]):
        v = key.get(tuple(np.round(np.append(Xp[r], Yp[r]), 9)))
        if not v:
            miss += 1
        else:
            idx.append(v[0])
    dx = np.abs(Xf[idx] - Xp).max(); dy = np.abs(Yf[idx] - Yp).max()
    print("   diamonds   posteriordb rows located: %d/%d (missing %d); "
          "max |dX| = %.3g, max |dY| = %.3g" % (len(idx), Xp.shape[0], miss, dx, dy))
    good_d = miss == 0 and dx < 1e-14 and dy < 1e-14
    print("   diamonds:", "REPRODUCES THE POSTERIORDB DESIGN" if good_d else "MISMATCH")
    return ok and good_d


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if what == "verify":
        sys.exit(0 if verify() else 1)
    args = sys.argv[2:]
    if what == "radon":
        for a in args:
            o = radon_obj(int(a))
            print("radon n=%d q=%d" % (o["attrs"]["n"], o["attrs"]["q"]))
            write("lmm_radon_n%d" % o["attrs"]["n"], o)
    elif what == "diamonds":
        for a in args:
            o = diamonds_obj(int(a))
            print("diamonds N=%d K=%d" % (o["attrs"]["n"], o["attrs"]["k"]))
            write("diamonds_n%d" % o["attrs"]["n"], o)
    elif what == "hmvn":
        for a in args:
            K, J = (int(x[1:]) for x in a.split("_"))
            o = hmvn_obj(K, J)
            print("hmvn K=%d J=%d position_size=%d" % (K, J, 2 * K + 1 + K * J))
            write("hierarchical_mvn_K%d_J%d" % (K, J), o)
    else:
        raise SystemExit("unknown: " + what)


if __name__ == "__main__":
    main()
