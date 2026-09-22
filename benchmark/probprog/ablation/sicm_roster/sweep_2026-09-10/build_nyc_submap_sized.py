#!/usr/bin/env python3
"""Build NYC traffic-accident CAR sub-maps at additional sizes for the
2026-09-10 problem-size sweep, and the harness-format data files that go with
them.

Source and sub-map rule are IDENTICAL to build_nyc_submap.py (which produced
the n=250 row of figure 10): posteriordb data set `traffic_accident_nyc`
(posterior `traffic_accident_nyc-bym2_offset_only`), file
posterior_database/data/data/traffic_accident_nyc.json.zip, N=1921 census
tracts, N_edges=5461 undirected adjacency edges as 1-based node1/node2 lists,
observed accident counts y and exposure E.

  * n < N : deterministic breadth-first search from tract 0 (0-based), each
            frontier node's neighbours expanded in ascending node-id order,
            stopped at n distinct tracts; the collected ids are sorted
            ascending and y, E and the adjacency are restricted to the induced
            subgraph. No randomness, no seed.
  * n = N : the whole map, every tract and every recorded edge. No sub-map rule
            applies.

Two files are written per size:
  1. pplbench/pplbench/models/real_data/nyc_traffic_submap_n<N>.json
     (same schema as nyc_traffic_submap.json)
  2. ablation/sicm_roster/data/car_spatial_nyc_n<N>.json
     the harness-format data file, byte-for-byte the structure that
     pplbench's SubprocessInference._serialize_data emits: every data_var as a
     nested list, plus coords / dims / attrs / init_params. Q_inv, log_E and Y
     come from models/car_spatial.py's `nyc_traffic` branch run on the new
     sub-map file; init_params from CARSpatial.generate_init_params(1396133,
     **attrs), which is the trial seed pplbench derives from the config's model
     seed 12345 (np.random.RandomState(12345): compile seed 7735426, then trial
     seeds 1396133, 2993578, ...) and the seed run_frozen_batch.sh uses.

Usage: build_nyc_submap_sized.py <n> [<n> ...]
"""
import collections
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROSTER = os.path.normpath(os.path.join(HERE, ".."))
BENCH = os.path.normpath(os.path.join(ROSTER, "..", ".."))
SRC = ("/mnt/sbrantq/probprog/posteriordb/posterior_database/data/data/"
       "traffic_accident_nyc.json.zip")
REAL = os.path.join(BENCH, "pplbench", "pplbench", "models", "real_data")
DATA = os.path.join(ROSTER, "data")
SEED_TRACT = 0
TRIAL_SEED = 1396133
RHO = 0.99


def load_source():
    z = zipfile.ZipFile(SRC)
    raw = json.loads(z.read("traffic_accident_nyc.json"))
    return raw


def submap(raw, n):
    N = int(raw["N"])
    node1 = np.asarray(raw["node1"], dtype=np.int64) - 1
    node2 = np.asarray(raw["node2"], dtype=np.int64) - 1
    y_full = np.asarray(raw["y"], dtype=np.float64)
    E_full = np.asarray(raw["E"], dtype=np.float64)
    assert len(node1) == int(raw["N_edges"]) == 5461, (len(node1), raw["N_edges"])

    adj = collections.defaultdict(set)
    for a, b in zip(node1, node2):
        if a == b:
            continue
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    if n >= N:
        idx = np.arange(N)
        rule = "whole map (no sub-map rule; every tract and every recorded edge)"
    else:
        seen = {SEED_TRACT}
        order = [SEED_TRACT]
        q = collections.deque([SEED_TRACT])
        while q and len(order) < n:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    order.append(v)
                    q.append(v)
                    if len(order) >= n:
                        break
        assert len(order) == n, (
            "BFS exhausted the component at %d tracts (asked for %d)"
            % (len(order), n))
        idx = np.sort(np.array(order))
        rule = ("deterministic breadth-first search from tract %d (0-based), "
                "neighbours expanded in ascending node-id order, stopped at %d "
                "distinct tracts; ids sorted ascending, induced subgraph kept"
                % (SEED_TRACT, n))

    pos = {int(t): i for i, t in enumerate(idx)}
    m = len(idx)
    W = np.zeros((m, m))
    for a, b in zip(node1, node2):
        a, b = int(a), int(b)
        if a in pos and b in pos and a != b:
            W[pos[a], pos[b]] = 1.0
            W[pos[b], pos[a]] = 1.0
    return idx, W, y_full[idx], E_full[idx], rule, N, int(raw["N_edges"])


def car_dataset(W, y, E, rho=RHO):
    """models/car_spatial.py, `nyc_traffic` branch, verbatim arithmetic."""
    n = W.shape[0]
    D = np.diag(np.sum(W, axis=1))
    Q = rho * (D - W) + (1 - rho) * np.eye(n)
    Q_inv = np.linalg.inv(Q)
    Q_inv = 0.5 * (Q_inv + Q_inv.T)
    Q_inv += 1e-6 * np.eye(n)
    log_E = np.log(E)
    return Q_inv, log_E, y


def init_params(n, seed=TRIAL_SEED):
    """CARSpatial.generate_init_params, verbatim."""
    rng = np.random.default_rng(seed)
    return {
        "alpha": rng.uniform(-1, 1, size=(1,)).astype(np.float64),
        "sigma_phi": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
        "phi": rng.standard_normal(n).astype(np.float64) * 0.1,
    }


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [500, 1000, 1921]
    raw = load_source()
    for n in sizes:
        idx, W, y, E, rule, N, N_edges = submap(raw, n)
        m = W.shape[0]
        deg = W.sum(1)
        edges = int(W.sum() // 2)
        Q_inv, log_E, Y = car_dataset(W, y, E)
        w = np.linalg.eigvalsh(
            RHO * (np.diag(deg) - W) + (1 - RHO) * np.eye(m))
        chol_ok = True
        try:
            np.linalg.cholesky(Q_inv)
        except np.linalg.LinAlgError:
            chol_ok = False
        print("n=%d: induced undirected edges=%d, degree min/med/max=%d/%d/%d, "
              "isolated=%d, cond(Q)=%.4g, chol(Q_inv) ok=%s"
              % (m, edges, deg.min(), np.median(deg), deg.max(),
                 int((deg == 0).sum()), w.max() / w.min(), chol_ok))
        print("   y: sum=%d min=%d max=%d ; E min=%.4g max=%.4g ; sum(y)/sum(E)=%.5f"
              % (y.sum(), y.min(), y.max(), E.min(), E.max(), y.sum() / E.sum()))

        prov = (
            "NYC traffic-accident CAR data, %s (%d tracts, %d induced undirected "
            "edges). Source: posteriordb data set `traffic_accident_nyc` "
            "(posterior `traffic_accident_nyc-bym2_offset_only`), file "
            "posterior_database/data/data/traffic_accident_nyc.json.zip from the "
            "local posteriordb checkout at /mnt/sbrantq/probprog/posteriordb, "
            "which carries N=%d tracts, N_edges=%d undirected adjacency edges "
            "(1-based node1/node2 lists), observed accident counts y and "
            "exposure E. Sub-map rule: %s. y and E are the observed values, "
            "unmodified; nothing is simulated. Same source and same rule as "
            "nyc_traffic_submap.json (the n=250 row of the SICM ablation "
            "figure), only the stopping size differs. Built by "
            "benchmark/probprog/ablation/sicm_roster/sweep_2026-09-10/"
            "build_nyc_submap_sized.py on 2026-09-10."
            % ("whole map" if m >= N else "contiguous sub-map", m, edges,
               N, N_edges, rule))

        real_path = os.path.join(REAL, "nyc_traffic_submap_n%d.json" % m)
        with open(real_path, "w") as f:
            json.dump({"_provenance": prov, "N": m,
                       "tract_index": [int(t) for t in idx],
                       "y": y.tolist(), "E": E.tolist(), "W": W.tolist()}, f)
        print("   wrote", real_path, "(%.1f MB)" % (os.path.getsize(real_path) / 1e6))

        ip = init_params(m)
        harness = {
            "_provenance": prov + (
                " Harness-format file: Q_inv, log_E and Y are produced by "
                "models/car_spatial.py's `nyc_traffic` branch arithmetic "
                "(Q = rho (D - W) + (1 - rho) I with rho=%.2f, Q_inv symmetrised "
                "and jittered by 1e-6 I, log_E = log E), serialised in the exact "
                "shape pplbench's SubprocessInference._serialize_data emits. "
                "init_params = CARSpatial.generate_init_params(%d, n=%d), the "
                "trial seed pplbench derives from the ablation config's model "
                "seed 12345 and the seed run_frozen_batch.sh uses."
                % (RHO, TRIAL_SEED, m)),
            "Q_inv": Q_inv.tolist(),
            "log_E": log_E.tolist(),
            "Y": Y.tolist(),
            "coords": {"i": list(range(m)), "j": list(range(m)),
                       "area": list(range(m))},
            "dims": {"Q_inv": ["i", "j"], "log_E": ["area"], "Y": ["area"]},
            "attrs": {"n": m, "rho": RHO},
            "init_params": {k: v.tolist() for k, v in ip.items()},
        }
        data_path = os.path.join(DATA, "car_spatial_nyc_n%d.json" % m)
        with open(data_path, "w") as f:
            json.dump(harness, f)
        print("   wrote", data_path, "(%.1f MB)" % (os.path.getsize(data_path) / 1e6))


if __name__ == "__main__":
    main()
