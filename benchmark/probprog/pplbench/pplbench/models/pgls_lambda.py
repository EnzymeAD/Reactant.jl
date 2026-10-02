# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Phylogenetic generalized least squares with Pagel's lambda ESTIMATED, on the
caper `shorebird` data:

    y      ~ MVN(X beta, sigma^2 (lambda C + (1 - lambda) I))
    beta   ~ Normal(0, 10)^k
    sigma  ~ HalfNormal(1)
    lambda ~ Uniform(0, 1)

C is the Brownian-motion correlation of the known tree (data). The tree is
ultrametric and C has unit diagonal to within 3.7e-8, so lambda C + (1 - lambda) I
is Pagel's lambda transform (Pagel 1999; Freckleton, Harvey and Pagel 2002),
and the model is the Bayesian form of the caper vignette's
`pgls(log(Egg.Mass) ~ log(M.Mass), shorebird, lambda = 'ML')`. The fixed-lambda
companion is `pgls.py` (lambda = 1). Same data file, `shorebird_pgls.json`.

SICM: CholeskyEigenLift on chol(sigma^2 lambda C + sigma^2 (1 - lambda) I), with
the whitening solves of y and X and the diagonal extraction rewritten.
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


class PGLSLambda(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        # beta and sigma drawn exactly as PGLS.generate_init_params, lambda last.
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 2))
        out = {
            "beta": rng.uniform(-1, 1, size=(k,)).astype(np.float64),
            "sigma": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
        }
        out["lambda"] = rng.uniform(0.1, 0.9, size=(1,)).astype(np.float64)
        return out

    @staticmethod
    def generate_data(
        seed: int, n: int = 71, dataset: str = "shorebird", train_frac: float = 1.0
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        assert dataset == "shorebird", dataset
        with open(os.path.join(REAL_DATA_DIR, "shorebird_pgls.json")) as f:
            raw = json.load(f)
        n, k = int(raw["N"]), int(raw["K"])
        C = np.asarray(raw["C"], dtype=np.float64)
        X = np.asarray(raw["X"], dtype=np.float64)
        y = np.asarray(raw["y"], dtype=np.float64)
        LOGGER.info(
            f"pgls_lambda[{dataset}]: n={n}, k={k}, "
            f"max|diag C - 1|={np.abs(np.diag(C) - 1).max():.3g}"
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
        beta = np.asarray(samples.beta.values)
        sigma = np.asarray(samples.sigma.values).reshape(-1)
        lam = np.asarray(samples["lambda"].values).reshape(-1)
        pll = np.zeros(len(sigma))
        for i in range(len(sigma)):
            S = sigma[i] ** 2 * (lam[i] * C + (1 - lam[i]) * np.eye(n))
            L = np.linalg.cholesky(S)
            z = np.linalg.solve(L, y - X @ beta[i])
            pll[i] = (-0.5 * float(z @ z) - float(np.sum(np.log(np.diag(L))))
                      - 0.5 * n * np.log(2 * np.pi))
        return pll
