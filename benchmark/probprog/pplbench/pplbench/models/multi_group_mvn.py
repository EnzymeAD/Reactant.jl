# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class MultiGroupMVN(BaseModel):
    """
    Multi-Group MVN: K groups share sigma^2 * R covariance.
    TriSolveScale dominates with K >> 1.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        return {"sigma": np.array([1.0])}

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 100,
        K: int = 10,
        lengthscale: float = 3.0,
        true_sigma: float = 1.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Correlation matrix (SE kernel)
        x = np.linspace(0, 10, n)
        delta = x[:, None] - x[None, :]
        R = np.exp(-0.5 * (delta / lengthscale) ** 2)
        R += 1e-6 * np.eye(n)  # jitter
        L = np.linalg.cholesky(R)

        # Group means
        mu = rng.standard_normal((K, n)) * 2.0

        # Observations
        Y = np.zeros((K, n))
        for k in range(K):
            Y[k] = mu[k] + true_sigma * (L @ rng.standard_normal(n))

        data_vars = {
            "R": (["i", "j"], R),
            "mu": (["group", "point"], mu),
            "Y": (["group", "point"], Y),
        }

        data = xr.Dataset(
            data_vars,
            coords={"group": np.arange(K), "point": np.arange(n)},
            attrs={"n": n, "K": K},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        sigma_samples = samples.sigma.values
        n_draws = len(sigma_samples)
        pll = np.zeros(n_draws)
        # Simplified: just return zeros
        return pll
