# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _se_kernel(x, rho):
    """Squared exponential kernel (no noise, no jitter)."""
    delta = x[:, None] - x[None, :]
    return np.exp(-0.5 * (delta / rho) ** 2)


class GPFixedLengthscale(BaseModel):
    """
    GP regression with fixed length scale (empirical Bayes).

    y ~ MVN(0, alpha^2 * K_base + sigma^2 * I)
      alpha ~ HalfNormal(1)
      sigma ~ HalfNormal(1)
      K_base = SE(X; rho_fixed)

    SICM: CholeskyEigenLift hoists eigendecomposition of K_base.
    O(N^3) -> O(N^2) per iteration.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "alpha": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "sigma": np.exp(rng.uniform(-2, 0, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 60,
        rho: float = 2.0,
        true_alpha: float = 1.5,
        true_sigma: float = 0.3,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        x = np.linspace(-5, 5, n)
        K_base = _se_kernel(x, rho)
        cov = true_alpha ** 2 * K_base + true_sigma ** 2 * np.eye(n)
        L = np.linalg.cholesky(cov)
        y = L @ rng.standard_normal(n)

        data = xr.Dataset(
            {"X": (["point"], x), "Y": (["point"], y)},
            coords={"point": np.arange(n)},
            attrs={"n": n, "rho": rho},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        x = test.X.values
        y = test.Y.values
        n = len(y)
        rho = test.attrs["rho"]

        alpha_samples = samples.alpha.values
        sigma_samples = samples.sigma.values
        n_draws = len(alpha_samples)

        K_base = _se_kernel(x, rho)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            a = np.abs(alpha_samples[i])
            s = np.abs(sigma_samples[i])
            cov = a ** 2 * K_base + s ** 2 * np.eye(n)
            try:
                L = np.linalg.cholesky(cov)
                v = np.linalg.solve(L, y)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                pll[i] = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
