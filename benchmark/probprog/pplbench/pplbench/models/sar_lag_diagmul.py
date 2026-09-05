# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Bayesian spatial-lag (SAR) model on the Boston housing data.

    (I - rho W) y = X beta + eps,   eps ~ N(0, sigma^2 I)

with W a known contiguity matrix (fixed data). Provenance, the two weight
matrices and the determinant identity that lets the log-determinant be taken as
a Cholesky: `real_data/README.md` and
`ablation/sicm_roster/build_boston_sar.py`.

SICM: DotScale on the lag term `(rho W) y` (W and y are both data, so `W y`
hoists) plus CholeskyEigenLift on the log-determinant of I - rho Ws.
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

DATASETS = {"boston": "boston_sar.json"}


def _rho_of(rho_raw, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-rho_raw))


class SARLag(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 14))
        return {
            "rho_raw": rng.uniform(-0.5, 0.5, size=(1,)).astype(np.float64),
            "beta": rng.uniform(-0.5, 0.5, size=(k,)).astype(np.float64),
            "sigma": np.exp(rng.uniform(-1, 0.0, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int, n: int = 506, dataset: str = "boston", train_frac: float = 1.0
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        # Real data: `seed` is unused, nothing is simulated. `n` is required by
        # the config schema and is cross-checked against the data file rather
        # than used to size anything.
        with open(os.path.join(REAL_DATA_DIR, DATASETS[dataset])) as f:
            raw = json.load(f)
        n, k = int(raw["N"]), int(raw["K"])
        W = np.asarray(raw["W"], dtype=np.float64)
        Ws = np.asarray(raw["Ws"], dtype=np.float64)
        X = np.asarray(raw["X"], dtype=np.float64)
        y = np.asarray(raw["y"], dtype=np.float64)
        lo, hi = float(raw["rho_lo"]), float(raw["rho_hi"])
        LOGGER.info(
            f"sar_lag[{dataset}]: n={n}, k={k}, rho in ({lo:.4f}, {hi:.4f}), "
            f"profile rho_hat={raw['profile_rho_hat']:.4f}, "
            f"sigma_hat={raw['profile_sigma_hat']:.4f}"
        )
        data = xr.Dataset(
            {
                "W": (["i", "j"], W),
                "Ws": (["i", "j"], Ws),
                "X": (["obs", "feature"], X),
                "Y": (["obs"], y),
            },
            coords={
                "i": np.arange(n),
                "j": np.arange(n),
                "obs": np.arange(n),
                "feature": np.arange(k),
            },
            attrs={"n": n, "k": k, "rho_lo": lo, "rho_hi": hi},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        W = test.W.values
        Ws = test.Ws.values
        X = test.X.values
        y = test.Y.values
        n = len(y)
        lo = float(test.attrs["rho_lo"])
        hi = float(test.attrs["rho_hi"])

        beta = samples.beta.values
        sigma = np.abs(np.atleast_1d(samples.sigma.values).reshape(-1))
        rho_raw = np.atleast_1d(samples.rho_raw.values).reshape(-1)
        eye = np.eye(n)

        pll = np.zeros(len(sigma))
        for i in range(len(sigma)):
            rho = _rho_of(rho_raw[i], lo, hi)
            Ay = y - rho * (W @ y)
            r = Ay - X @ beta[i]
            s2 = sigma[i] ** 2
            sign, logdet = np.linalg.slogdet(eye - rho * Ws)
            if sign <= 0:
                pll[i] = -1e10
                continue
            pll[i] = (logdet - 0.5 * float(r @ r) / s2
                      - 0.5 * n * np.log(2 * np.pi * s2))
        return pll
