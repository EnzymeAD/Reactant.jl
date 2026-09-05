# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Phylogenetic generalized least squares on the caper `shorebird` data.

    y ~ MVN(X beta, sigma^2 C)

with C the Brownian-motion phylogenetic correlation of a known tree, i.e. fixed
data. Mirrors the caper vignette's `pgls(log(Egg.Mass) ~ log(M.Mass), shorebird)`
at lambda = 1. Provenance and the tree -> C construction:
`real_data/README.md` and `ablation/sicm_roster/build_shorebird_pgls.py`.

SICM: the TriangularSolveScale cascade. The density whitens y and X separately
against chol(sigma^2 C), so both solve right-hand sides are data and the solves
themselves hoist.
"""
import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "real_data"
)


class PGLS(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 2))
        return {
            "beta": rng.uniform(-1, 1, size=(k,)).astype(np.float64),
            "sigma": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int, n: int = 71, dataset: str = "shorebird", train_frac: float = 1.0
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        # Real data: `seed` is unused, nothing is simulated. `n` is required by
        # the config schema and is cross-checked against the data file rather
        # than used to size anything.
        assert dataset == "shorebird", dataset
        with open(os.path.join(REAL_DATA_DIR, "shorebird_pgls.json")) as f:
            raw = json.load(f)
        n, k = int(raw["N"]), int(raw["K"])
        C = np.asarray(raw["C"], dtype=np.float64)
        X = np.asarray(raw["X"], dtype=np.float64)
        y = np.asarray(raw["y"], dtype=np.float64)
        w = np.linalg.eigvalsh(C)
        LOGGER.info(
            f"pgls[{dataset}]: n={n}, k={k}, cond(C)={w.max() / w.min():.4g}, "
            f"GLS reference beta={np.round(raw['gls_beta_reference'], 4).tolist()}"
        )
        data = xr.Dataset(
            {
                "C": (["i", "j"], C),
                "X": (["obs", "feature"], X),
                "Y": (["obs"], y),
            },
            coords={
                "i": np.arange(n),
                "j": np.arange(n),
                "obs": np.arange(n),
                "feature": np.arange(k),
            },
            attrs={"n": n, "k": k},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        C = test.C.values
        X = test.X.values
        y = test.Y.values
        n = len(y)
        L = np.linalg.cholesky(C)
        log_det_C = 2.0 * float(np.sum(np.log(np.diag(L))))

        beta = samples.beta.values
        sigma = np.abs(np.atleast_1d(samples.sigma.values).reshape(-1))
        pll = np.zeros(len(sigma))
        for i in range(len(sigma)):
            r = y - X @ beta[i]
            z = np.linalg.solve(L, r)
            s2 = sigma[i] ** 2
            pll[i] = -0.5 * (float(z @ z) / s2 + log_det_C
                             + n * np.log(2 * np.pi * s2))
        return pll
