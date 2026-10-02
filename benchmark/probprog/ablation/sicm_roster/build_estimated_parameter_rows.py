#!/usr/bin/env python3
"""Build the harness-format data files of the three estimated-parameter SICM
rows (2026-09-24) and validate them against the fixed-parameter rows' files.

Rows and files written (ablation/sicm_roster/data/):
  car_leroux_scotland.json   models/car_leroux.py, dataset="scotland", n=56
  car_leroux_nyc_n1921.json  models/car_leroux.py, dataset="nyc_traffic_n1921"
  pgls_lambda.json           models/pgls_lambda.py, dataset="shorebird", n=71

Each file is the model class's own `generate_data` output serialised in the
exact shape pplbench's SubprocessInference._serialize_data emits, with
init_params = <Model>.generate_init_params(1396133, **attrs): 1396133 is the
first trial seed pplbench derives from the ablation config's model seed 12345
and the seed run_frozen_batch.sh uses (FROZEN_SEED).

Validation against the fixed-parameter file of the same data (the files the
figure-10 rows measure):
  * log_E, Y (CAR) and C, X, Y (PGLS) bit-identical;
  * CAR: the fixed-rho file's Q_inv is recomputed from the new file's R with the
    car_spatial.py arithmetic (Q = 0.99 R + 0.01 I, inverse, symmetrised,
    + 1e-6 I) and must be bit-identical (Scotland) or equal to rounding
    (NYC n=1921, where a dense 1921 x 1921 inverse is not bit-reproducible
    across BLAS thread counts); for NYC the adjacency, y and log E are in
    addition checked bit-identical against the posteriordb source through the
    size sweep's own builder (sweep_2026-09-10/build_nyc_submap_sized.py);
  * init_params of the shared sites (alpha, sigma_phi, phi / beta, sigma)
    bit-identical to the fixed-parameter file's.

Usage: build_estimated_parameter_rows.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(BENCH, "pplbench"))
from pplbench.models.car_leroux import CARLeroux  # noqa: E402
from pplbench.models.pgls_lambda import PGLSLambda  # noqa: E402

TRIAL_SEED = 1396133
MODEL_SEED = 12345


def serialize(data, init_params):
    """pplbench SubprocessInference._serialize_data, verbatim."""
    result = {}
    for var in data.data_vars:
        result[var] = data[var].values.tolist()
    result["coords"] = {c: data.coords[c].values.tolist() for c in data.coords}
    result["dims"] = {var: list(data[var].dims) for var in data.data_vars}
    result["attrs"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in data.attrs.items()}
    result["init_params"] = ({k: v.tolist() for k, v in init_params.items()}
                             if init_params else None)
    return result


def same(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a, b)


def check(name, ok):
    print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        raise SystemExit(f"validation failed: {name}")


def components(R):
    W = (np.diag(np.diag(R)) - R) != 0
    n = len(R)
    seen = np.zeros(n, bool)
    c = 0
    for s in range(n):
        if seen[s]:
            continue
        c += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(W[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return c


def car_row(dataset, n, key, fixed_file):
    print(f"{key}: CARLeroux.generate_data(dataset={dataset!r}, n={n})")
    data, _ = CARLeroux.generate_data(MODEL_SEED, n=n, dataset=dataset)
    ip = CARLeroux.generate_init_params(TRIAL_SEED, **data.attrs)
    out = serialize(data, ip)
    fixed = json.load(open(os.path.join(DATA, fixed_file)))

    R = np.asarray(out["R"])
    check("log_E bit-identical to " + fixed_file, same(out["log_E"], fixed["log_E"]))
    check("Y bit-identical to " + fixed_file, same(out["Y"], fixed["Y"]))
    rho = float(fixed["attrs"]["rho"])
    Q = rho * R + (1 - rho) * np.eye(n)
    Q_inv = np.linalg.inv(Q)
    Q_inv = 0.5 * (Q_inv + Q_inv.T)
    Q_inv += 1e-6 * np.eye(n)
    Qf = np.asarray(fixed["Q_inv"])
    if same(Q_inv, Qf):
        check("Q_inv recomputed from R bit-identical to " + fixed_file, True)
    else:
        # A dense inverse is not bit-reproducible across BLAS thread counts;
        # report the rounding-level difference and check the adjacency exactly
        # below instead.
        rel = float(np.max(np.abs(Q_inv - Qf)) / np.max(np.abs(Qf)))
        check(f"Q_inv recomputed from R equal to {fixed_file} to rounding "
              f"(max rel diff {rel:.2e} < 1e-10)", rel < 1e-10)
    if dataset.startswith("nyc"):
        sys.path.insert(0, os.path.join(HERE, "sweep_2026-09-10"))
        import build_nyc_submap_sized as nb
        _, W_src, y_src, E_src, _, _, _ = nb.submap(nb.load_source(), n)
        check("W = diag(R) - R bit-identical to the posteriordb "
              "traffic_accident_nyc adjacency (whole map, 5461 edges)",
              same(np.diag(np.diag(R)) - R, W_src)
              and int(W_src.sum() // 2) == 5461)
        check("Y, log_E bit-identical to posteriordb y, log(E)",
              same(out["Y"], y_src) and same(out["log_E"], np.log(E_src)))
    for k in ("alpha", "sigma_phi", "phi"):
        check(f"init_params[{k}] bit-identical to " + fixed_file,
              same(out["init_params"][k], fixed["init_params"][k]))
    lam = np.linalg.eigvalsh(R)
    deg = np.diag(R)
    print(f"   n={n} edges={int(-(R - np.diag(deg)).sum() // 2)} "
          f"degree min/median/max={deg.min():.0f}/{np.median(deg):.0f}/{deg.max():.0f} "
          f"components={components(R)} eig(R) in [{lam[0]:.3e}, {lam[-1]:.4g}] "
          f"init rho={out['init_params']['rho'][0]:.6f}")
    out["_provenance"] = (
        f"Leroux CAR with rho estimated (models/car_leroux.py, dataset="
        f"{dataset!r}), R = D - W of the real adjacency, log_E = log E, Y = y "
        f"(observed, unmodified). Built by ablation/sicm_roster/"
        f"build_estimated_parameter_rows.py on 2026-09-24; log_E and Y validated "
        f"bit-identical against data/{fixed_file}, the adjacency through the "
        f"fixed-rho Q_inv recomputed from R (see build_data.log in "
        f"oopsla/evidence/estimated_parameter_rows_2026-09-24). init_params = CARLeroux.generate_init_params("
        f"{TRIAL_SEED}, n={n}); alpha, sigma_phi, phi equal the fixed-rho row's.")
    return out


def pgls_row():
    key = "pgls_lambda"
    print(f"{key}: PGLSLambda.generate_data(dataset='shorebird', n=71)")
    data, _ = PGLSLambda.generate_data(MODEL_SEED, n=71, dataset="shorebird")
    ip = PGLSLambda.generate_init_params(TRIAL_SEED, **data.attrs)
    out = serialize(data, ip)
    fixed = json.load(open(os.path.join(DATA, "pgls.json")))
    for k in ("C", "X", "Y"):
        check(f"{k} bit-identical to pgls.json", same(out[k], fixed[k]))
    for k in ("beta", "sigma"):
        check(f"init_params[{k}] bit-identical to pgls.json",
              same(out["init_params"][k], fixed["init_params"][k]))
    C = np.asarray(out["C"])
    w = np.linalg.eigvalsh(C)
    print(f"   n=71 k=2 max|diag C - 1|={np.abs(np.diag(C) - 1).max():.3e} "
          f"eig(C) in [{w[0]:.4g}, {w[-1]:.4g}] "
          f"init lambda={out['init_params']['lambda'][0]:.6f}")
    out["_provenance"] = (
        "PGLS with Pagel's lambda estimated (models/pgls_lambda.py), caper "
        "shorebird data (real_data/shorebird_pgls.json). Built by "
        "ablation/sicm_roster/build_estimated_parameter_rows.py on 2026-09-24; "
        "C, X, Y validated bit-identical against data/pgls.json. init_params = "
        f"PGLSLambda.generate_init_params({TRIAL_SEED}, n=71, k=2); beta, sigma "
        "equal the fixed-lambda row's.")
    return key, out


def main():
    rows = [("car_leroux_scotland",
             car_row("scotland", 56, "car_leroux_scotland",
                     "car_spatial_scotland.json")),
            ("car_leroux_nyc_n1921",
             car_row("nyc_traffic_n1921", 1921, "car_leroux_nyc_n1921",
                     "car_spatial_nyc_n1921.json")),
            pgls_row()]
    for key, out in rows:
        path = os.path.join(DATA, key + ".json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
