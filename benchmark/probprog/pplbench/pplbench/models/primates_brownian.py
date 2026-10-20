# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Brownian-motion phylogenetic regression of primate brain size (section 2).

McElreath, Statistical Rethinking, 2nd ed., section 14.5.2, R code 14.47-14.51,
model m14.10, on Primates301 (n = 151 complete cases):

    B ~ multi_normal(mu, SIGMA);  mu <- a + bM*M + bG*G;  SIGMA <- R * sigma_sq
    a ~ normal(0, 1);  c(bM, bG) ~ normal(0, 0.5);  sigma_sq ~ exponential(1)

R is the Brownian-motion correlation of the primate phylogeny, i.e. fixed data.
The data file is the one the Impulse program reads
(`ablation/sicm_roster/data/brownian_primates_n151.json`, X = [1, M, G], Y = B).

Inits follow the data file's convention (numpy default_rng(seed); a, b = (bM, bG)
~ U(-1, 1); sigma_sq = exp(U(-1, 0.5))), so trial seed 1396133 reproduces the
data file's init_params. The slopes are given twice: as `b` (the Impulse
program's convention) and as `bM`, `bG` (the Stan parameter names). Each system
reads the names it declares, so every system starts a trial from the same point.
"""
import json
import os
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "ablation", "sicm_roster", "data", "brownian_primates_n151.json",
)


class PrimatesBrownian(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        a = rng.uniform(-1, 1, size=(1,))
        b = rng.uniform(-1, 1, size=(2,))
        sigma_sq = np.exp(rng.uniform(-1, 0.5, size=(1,)))
        return {"a": a, "b": b, "bM": b[0:1], "bG": b[1:2], "sigma_sq": sigma_sq}

    @staticmethod
    def generate_data(seed: int, n: int = 151) -> Tuple[xr.Dataset, xr.Dataset]:
        # Real data: `seed` is unused and nothing is simulated. `n` is checked
        # against the data file.
        with open(DATA_FILE) as f:
            raw = json.load(f)
        assert int(raw["attrs"]["n"]) == n, (raw["attrs"]["n"], n)
        R = np.asarray(raw["R"], dtype=np.float64)
        X = np.asarray(raw["X"], dtype=np.float64)
        Y = np.asarray(raw["Y"], dtype=np.float64)
        data = xr.Dataset(
            {
                "R": (["species", "species2"], R),
                "X": (["species", "coef"], X),
                "Y": (["species"], Y),
            },
            coords={
                "species": np.arange(n),
                "species2": np.arange(n),
                "coef": np.arange(X.shape[1]),
            },
            attrs={"n": n, "k": int(X.shape[1])},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        R = test.R.values
        X = test.X.values
        y = test.Y.values
        n = len(y)
        L = np.linalg.cholesky(R)
        log_det_R = 2.0 * float(np.sum(np.log(np.diag(L))))
        a = np.atleast_1d(samples.a.values).reshape(-1)
        b = np.asarray(samples.b.values).reshape(len(a), 2)
        s2 = np.atleast_1d(samples.sigma_sq.values).reshape(-1)
        mu = a[:, None] + b[:, 0:1] * X[None, :, 1] + b[:, 1:2] * X[None, :, 2]
        z = np.linalg.solve(L, (y[None, :] - mu).T)
        quad = np.sum(z * z, axis=0)
        return -0.5 * (quad / s2 + log_det_R + n * np.log(2 * np.pi * s2))
