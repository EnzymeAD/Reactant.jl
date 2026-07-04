# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class EigenLiftSolve(BaseModel):
    """EigenLift benchmark: chol(alpha^2*A + sigma^2*I) with tri_solve users."""

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 50))
        return {
            "alpha": np.array([1.0]),
            "sigma": np.array([0.5]),
            "f_tilde": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 50,
        true_alpha: float = 1.5,
        true_sigma: float = 0.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # PD base matrix
        V = rng.standard_normal((n, n)) * 0.1
        A_base = V.T @ V + 0.5 * np.eye(n)

        # 5 data vectors
        xs = [rng.standard_normal((n, 1)) for _ in range(5)]

        # True prediction
        K_true = true_alpha**2 * A_base + true_sigma**2 * np.eye(n)
        L_true = np.linalg.cholesky(K_true)
        pred = sum(np.linalg.solve(L_true, x) for x in xs).ravel()
        y = pred + rng.standard_normal(n)

        data_vars = {
            "A_base": (["i", "j"], A_base),
            "Y": (["point"], y),
        }
        for k in range(5):
            data_vars[f"x{k+1}"] = ([f"row_{k}", f"col_{k}"], xs[k])

        data = xr.Dataset(
            data_vars,
            coords={"point": np.arange(n)},
            attrs={"n": n},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        return np.zeros(len(samples.alpha.values))
