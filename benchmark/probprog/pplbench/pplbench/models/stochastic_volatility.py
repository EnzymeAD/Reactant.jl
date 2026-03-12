# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _ar1_covariance(T, phi):
    """Build AR(1) stationary covariance: C[s,t] = phi^|s-t| / (1 - phi^2)."""
    idx = np.arange(T)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.power(phi, diff) / (1.0 - phi ** 2)


class StochasticVolatility(BaseModel):
    """
    Stochastic Volatility: y_t ~ Normal(0, exp(h_t / 2))
    h = mu + sigma_h * L_C * eta  (NCP with AR(1) covariance)

    SICM: CholeskyScaleFactorization on sigma_h^2 * C_ar1
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        T = int(model_attrs.get("T", model_attrs.get("n", 100)))
        return {
            "mu": rng.uniform(-1, 1, size=(1,)).astype(np.float64),
            "sigma_h": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
            "eta": rng.standard_normal(T).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 100,
        phi: float = 0.95,
        true_mu: float = -1.0,
        true_sigma_h: float = 0.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)
        T = n

        # AR(1) stationary covariance
        C_ar1 = _ar1_covariance(T, phi)
        C_ar1 += 1e-6 * np.eye(T)

        # Generate log-volatility from MVN
        cov_h = true_sigma_h ** 2 * C_ar1
        L_h = np.linalg.cholesky(cov_h)
        h = true_mu + L_h @ rng.standard_normal(T)

        # Generate returns
        y = rng.standard_normal(T) * np.exp(h / 2.0)

        data = xr.Dataset(
            {
                "C_ar1": (["t1", "t2"], C_ar1),
                "Y": (["time"], y),
            },
            coords={
                "t1": np.arange(T),
                "t2": np.arange(T),
                "time": np.arange(T),
            },
            attrs={"n": T, "T": T, "phi": phi},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        y = test.Y.values
        T = len(y)

        mu_samples = samples.mu.values
        sigma_h_samples = samples.sigma_h.values
        eta_samples = samples.eta.values
        n_draws = len(mu_samples)

        C_ar1 = test.C_ar1.values

        pll = np.zeros(n_draws)
        L_C = np.linalg.cholesky(C_ar1)
        for i in range(n_draws):
            mu_i = mu_samples[i]
            sh_i = np.abs(sigma_h_samples[i])
            eta_i = eta_samples[i]
            h = mu_i + sh_i * (L_C @ eta_i)
            # log p(y|h) = sum_t Normal(y_t | 0, exp(h_t/2))
            scales = np.exp(h / 2.0)
            pll[i] = np.sum(
                -0.5 * np.log(2 * np.pi) - np.log(scales) - 0.5 * (y / scales) ** 2
            )

        return pll
