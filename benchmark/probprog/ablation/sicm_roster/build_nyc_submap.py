#!/usr/bin/env python3
"""Build the NYC traffic-accident contiguous sub-map used by the CAR NYC row.

Source: posteriordb data set `traffic_accident_nyc` (used by the posterior
`traffic_accident_nyc-bym2_offset_only`), shipped as
posterior_database/data/data/traffic_accident_nyc.json.zip. It carries
N = 1921 census tracts, N_edges = 5461 undirected adjacency edges given as
1-based (node1, node2) lists, observed accident counts y, and exposure E.

Sub-map rule (deterministic, no randomness): breadth-first search from tract 0
(0-based; node id 1 in the file's 1-based edge lists), expanding each frontier
node's neighbours in ascending node-id order, until SUBMAP_N distinct tracts
have been collected. The collected ids are then sorted ascending, so the
sub-map keeps the file's tract order, and y, E and the adjacency are restricted
to the induced subgraph on those tracts.

Writes pplbench/pplbench/models/real_data/nyc_traffic_submap.json with the same
field shapes as scotland_lip_cancer.json (N, y, E, W as a dense 0/1 matrix)
plus a _provenance string.
"""
import collections
import json
import os
import zipfile

import numpy as np

SRC = "/mnt4/sbrantq/probprog/posteriordb/posterior_database/data/data/traffic_accident_nyc.json.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "pplbench", "pplbench", "models", "real_data",
                   "nyc_traffic_submap.json")
SUBMAP_N = 250
SEED_TRACT = 0


def main():
    z = zipfile.ZipFile(SRC)
    raw = json.loads(z.read("traffic_accident_nyc.json"))
    N = int(raw["N"])
    node1 = np.asarray(raw["node1"], dtype=np.int64) - 1   # to 0-based
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

    # Deterministic BFS from SEED_TRACT, ascending neighbour order.
    seen = {SEED_TRACT}
    order = [SEED_TRACT]
    q = collections.deque([SEED_TRACT])
    while q and len(order) < SUBMAP_N:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                order.append(v)
                q.append(v)
                if len(order) >= SUBMAP_N:
                    break
    assert len(order) == SUBMAP_N, "BFS exhausted the component at %d tracts" % len(order)

    idx = np.sort(np.array(order))
    pos = {int(t): i for i, t in enumerate(idx)}
    n = len(idx)
    W = np.zeros((n, n))
    for a, b in zip(node1, node2):
        a, b = int(a), int(b)
        if a in pos and b in pos and a != b:
            W[pos[a], pos[b]] = 1.0
            W[pos[b], pos[a]] = 1.0
    y = y_full[idx]
    E = E_full[idx]

    deg = W.sum(1)
    edges = int(W.sum() // 2)
    print("sub-map: n=%d, induced undirected edges=%d, degree min/med/max=%d/%d/%d, "
          "isolated=%d" % (n, edges, deg.min(), np.median(deg), deg.max(),
                           int((deg == 0).sum())))
    print("y: sum=%d min=%d max=%d ; E: min=%d max=%d ; sum(y)/sum(E)=%.5f"
          % (y.sum(), y.min(), y.max(), E.min(), E.max(), y.sum() / E.sum()))
    for rho in (0.99, 0.95, 0.9):
        D = np.diag(deg)
        Q = rho * (D - W) + (1 - rho) * np.eye(n)
        w = np.linalg.eigvalsh(Q)
        Qi = np.linalg.inv(Q)
        Qi = 0.5 * (Qi + Qi.T) + 1e-6 * np.eye(n)
        ok = True
        try:
            np.linalg.cholesky(Qi)
        except np.linalg.LinAlgError:
            ok = False
        print("  rho=%.2f cond(Q)=%.4g eig(Q)=[%.4g, %.4g] chol(Q_inv) ok=%s"
              % (rho, w.max() / w.min(), w.min(), w.max(), ok))

    prov = (
        "NYC traffic-accident sub-map (%d census tracts). Source: posteriordb "
        "data set `traffic_accident_nyc` (posterior "
        "`traffic_accident_nyc-bym2_offset_only`), file "
        "posterior_database/data/data/traffic_accident_nyc.json.zip, which has "
        "N=%d tracts, N_edges=%d undirected adjacency edges (1-based node1/node2 "
        "lists), observed accident counts y and exposure E. Sub-map rule: "
        "deterministic breadth-first search from tract %d (0-based), expanding "
        "each frontier node's neighbours in ascending node-id order, until %d "
        "distinct tracts are collected; the collected ids are sorted ascending "
        "and y, E and the adjacency are restricted to the induced subgraph "
        "(%d undirected edges). y and E are the observed values, unmodified; "
        "nothing is simulated. Built by "
        "benchmark/probprog/ablation/sicm_roster/build_nyc_submap.py."
        % (n, N, int(raw["N_edges"]), SEED_TRACT, SUBMAP_N, edges)
    )
    out = {
        "_provenance": prov,
        "N": n,
        "tract_index": [int(t) for t in idx],
        "y": y.tolist(),
        "E": E.tolist(),
        "W": W.tolist(),
    }
    path = os.path.abspath(OUT)
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path, "(%.1f MB)" % (os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
