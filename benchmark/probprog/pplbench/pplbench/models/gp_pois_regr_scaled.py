# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _se_kernel(x, rho):
    """Squared exponential kernel matrix (no scaling, no jitter)."""
    delta = x[:, None] - x[None, :]
    return np.exp(-0.5 * (delta / rho) ** 2)


class GPPoisRegrScaled(BaseModel):
    """
    GP Poisson Regression with synthetic data at arbitrary N.

    Same model as gp_pois_regr (posteriordb) but generates synthetic data
    to evaluate SICM at larger problem sizes.

    Model:
        alpha ~ HalfNormal(2)
        f_tilde ~ Normal(0, 1)^N
        cov = alpha^2 * K_base (SE kernel, fixed rho)
        L = cholesky(cov)
        f = L @ f_tilde
        k ~ Poisson(exp(f))

    SICM: CholeskyScaleFactorization hoists O(N^3) Cholesky.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 50))
        return {
            "alpha": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "f_tilde": rng.uniform(-2, 2, size=(n,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 50,
        rho: float = 3.0,
        true_alpha: float = 1.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        x = np.linspace(-10, 10, n)
        K_base = _se_kernel(x, rho)
        cov = true_alpha ** 2 * K_base
        # Add small jitter for numerical stability during data generation
        cov += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(cov)
        f = L @ rng.standard_normal(n)
        k = rng.poisson(np.exp(np.clip(f, -10, 10))).astype(np.float64)

        data = xr.Dataset(
            {"X": (["point"], x), "Y": (["point"], k)},
            coords={"point": np.arange(n)},
            attrs={"n": n, "rho": rho},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        x = test.X.values
        k = test.Y.values
        n = len(x)
        rho = test.attrs["rho"]

        alpha_samples = samples.alpha.values
        f_tilde_samples = samples.f_tilde.values
        n_draws = len(alpha_samples)

        K_base = _se_kernel(x, rho)
        log_k_fact = gammaln(k + 1)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            alpha_i = np.atleast_1d(alpha_samples[i]).item()
            f_tilde_i = f_tilde_samples[i]
            cov = alpha_i ** 2 * K_base
            cov += 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(cov)
                f = L @ f_tilde_i
                pll[i] = np.sum(k * f - np.exp(f) - log_k_fact)
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
