# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""
Phylogenetic Poisson regression (motivating example).

Real data: the count-data example of the brms phylogenetics vignette (Buerkner),
from chapter 11 of *Modern Phylogenetic Comparative Methods* (de Villemereuil &
Nakagawa, 2014): 200 species on a known tree, a Poisson phenotype and one covariate.
See benchmark/probprog/motivating/phylo_data/README.md for provenance.

Model (the brms formula `phen_pois ~ cofactor + (1|gr(phylo, cov = A)) + (1|obs)`,
written in the natural non-centered form):

    sigma_sq  ~ Exponential(1)              (phylogenetic variance; cf. Statistical
                                             Rethinking m14.10: sigma_sq ~ exponential(1))
    eta       ~ Normal(0, 1)^n
    b0, b1    ~ Normal(0, 1)
    sigma_obs ~ HalfNormal(1)               (observation-level overdispersion scale)
    eps       ~ Normal(0, 1)^n
    L = cholesky(sigma_sq * A)              <- chol(s * A_fixed): the CholScale target
    f = L @ eta
    k ~ Poisson(exp(b0 + b1 * x + f + sigma_obs * eps))

A is `ape::vcv.phylo(tree)`: A[i, j] = shared root-to-MRCA path length of species i, j.
The tree is ultrametric with height 1, so diag(A) = 1 and A is a correlation matrix.
"""
import logging
import os
import re
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "motivating", "phylo_data"
)


def _parse_newick(newick: str):
    """Minimal Newick parser: returns nested dicts {name, len, children}."""
    pos = 0

    def node():
        nonlocal pos
        nd = {"children": [], "name": None, "len": 0.0}
        if newick[pos] == "(":
            pos += 1
            while True:
                nd["children"].append(node())
                if newick[pos] == ",":
                    pos += 1
                    continue
                if newick[pos] == ")":
                    pos += 1
                    break
        m = re.match(r"[^:,\)\(;]*", newick[pos:])
        nd["name"] = m.group(0)
        pos += len(m.group(0))
        if pos < len(newick) and newick[pos] == ":":
            pos += 1
            m = re.match(r"[0-9.eE+-]+", newick[pos:])
            nd["len"] = float(m.group(0))
            pos += len(m.group(0))
        return nd

    return node()


def vcv_from_nexus(path: str) -> Tuple[np.ndarray, list]:
    """ape::vcv.phylo for the (single) tree in a NEXUS file written by ape.

    Returns (A, tip_names) with A[i, j] = depth of the MRCA of tips i and j
    (= shared branch length from the root), A[i, i] = root-to-tip distance.
    """
    nex = open(path).read()
    translate = dict(re.findall(r"^\s*(\d+)\s+(\S+?),?\s*$", nex, re.M))
    tree_line = [l for l in nex.splitlines() if l.strip().upper().startswith("TREE")][0]
    newick = tree_line.split("=", 1)[1].strip()
    newick = re.sub(r"^\[&[RU]\]\s*", "", newick)
    root = _parse_newick(newick)

    tips = []  # (name, [depths of ancestors incl. self])

    def walk(nd, depth, path_ids, path_depths):
        d = depth + nd["len"]
        ids = path_ids + [id(nd)]
        ds = path_depths + [d]
        if not nd["children"]:
            tips.append((translate.get(nd["name"], nd["name"]), dict(zip(ids, ds))))
        for c in nd["children"]:
            walk(c, d, ids, ds)

    walk(root, 0.0, [], [])
    names = [t[0] for t in tips]
    n = len(tips)
    A = np.zeros((n, n))
    for i in range(n):
        anc_i = tips[i][1]
        for j in range(n):
            anc_j = tips[j][1]
            A[i, j] = max(anc_i[k] for k in anc_i if k in anc_j)
    return A, names


def load_phylo_pois(n: int = 200):
    """Load the vignette count data and the tree covariance, in data-row order."""
    A_full, names = vcv_from_nexus(os.path.join(DATA_DIR, "phylo.nex"))
    raw = open(os.path.join(DATA_DIR, "data_pois.txt")).read()
    rows = [r.split() for r in raw.replace('"', "").replace("\r", "\n").split("\n") if r.strip()]
    hdr = rows[0]
    species = [r[hdr.index("phylo")] for r in rows[1:]]
    k = np.array([float(r[hdr.index("phen_pois")]) for r in rows[1:]])
    z = np.array([float(r[hdr.index("cofactor")]) for r in rows[1:]])
    idx = [names.index(s) for s in species]
    A = A_full[np.ix_(idx, idx)]
    A = A / A.diagonal().max()  # unit diagonal (tree height 1: a no-op up to rounding)
    if n < len(k):  # optional subset: the first n species (a principal submatrix of A)
        A, k, z = A[:n, :n], k[:n], z[:n]
    x = (z - z.mean()) / z.std()  # standardized covariate
    return A, x, k


class PhyloPois(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 200))
        return {
            "sigma_sq": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "eta": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
            "b0": rng.uniform(-0.5, 0.5, size=(1,)).astype(np.float64),
            "b1": rng.uniform(-0.5, 0.5, size=(1,)).astype(np.float64),
            "sigma_obs": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "eps": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int, n: int = 200, train_frac: float = 1.0
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        # Real data: `seed` is unused (kept for the BaseModel interface).
        A, x, k = load_phylo_pois(n)
        n = len(k)
        w = np.linalg.eigvalsh(A)
        LOGGER.info(f"phylo_pois: n={n}, cond(A)={w.max() / w.min():.3g}, counts max={k.max():.0f}")
        data = xr.Dataset(
            {
                "A": (["i", "j"], A),
                "X": (["point"], x),
                "Y": (["point"], k),
            },
            coords={"i": np.arange(n), "j": np.arange(n), "point": np.arange(n)},
            attrs={"n": n},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        A = test.A.values
        x = test.X.values
        k = test.Y.values
        L = np.linalg.cholesky(A)
        log_k_fact = gammaln(k + 1)

        sigma_sq = np.atleast_1d(samples.sigma_sq.values).reshape(-1)
        b0 = np.atleast_1d(samples.b0.values).reshape(-1)
        b1 = np.atleast_1d(samples.b1.values).reshape(-1)
        sigma_obs = np.atleast_1d(samples.sigma_obs.values).reshape(-1)
        eta = samples.eta.values
        eps = samples.eps.values

        n_draws = len(sigma_sq)
        pll = np.zeros(n_draws)
        for i in range(n_draws):
            f = np.sqrt(sigma_sq[i]) * (L @ eta[i])
            log_rate = b0[i] + b1[i] * x + f + sigma_obs[i] * eps[i]
            pll[i] = np.sum(k * log_rate - np.exp(log_rate) - log_k_fact)
        return pll
