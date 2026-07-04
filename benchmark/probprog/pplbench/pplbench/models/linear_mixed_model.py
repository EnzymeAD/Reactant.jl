# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class LinearMixedModel(BaseModel):
    """
    Linear Mixed Model: y ~ MVN(X @ beta, sigma_u^2 * K_re + sigma_e^2 * I)

    Where K_re = Z @ Z' is precomputed from random effects design matrix Z.

    Parameters:
        beta ~ Normal(0, 1)^k       (fixed effects)
        sigma_u ~ HalfNormal(1)     (random effects std)
        sigma_e ~ HalfNormal(1)     (residual std)

    SICM: CholeskyEigenLift hoists eigendecomposition of K_re.
    O(N^3) -> O(N^2) per iteration.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 3))
        return {
            "beta": rng.uniform(-2, 2, size=(k,)).astype(np.float64),
            "sigma_u": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "sigma_e": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 100,
        k: int = 3,
        q: int = 20,
        true_sigma_u: float = 1.0,
        true_sigma_e: float = 0.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Fixed effects design
        X = rng.standard_normal((n, k))

        # Random effects design (e.g., group membership indicators)
        Z = np.zeros((n, q))
        group_assign = rng.integers(0, q, size=n)
        for i in range(n):
            Z[i, group_assign[i]] = 1.0

        # Precompute K_re = Z @ Z'
        K_re = Z @ Z.T

        # True parameters
        beta_true = rng.standard_normal(k)
        u_true = rng.standard_normal(q) * true_sigma_u

        # Generate response
        mu = X @ beta_true + Z @ u_true
        y = mu + rng.standard_normal(n) * true_sigma_e

        data = xr.Dataset(
            {
                "X": (["obs", "feature"], X),
                "K_re": (["i", "j"], K_re),
                "Y": (["obs"], y),
            },
            coords={
                "obs": np.arange(n),
                "feature": np.arange(k),
                "i": np.arange(n),
                "j": np.arange(n),
            },
            attrs={"n": n, "k": k, "q": q},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        X = test.X.values
        K_re = test.K_re.values
        y = test.Y.values
        n = len(y)

        beta_samples = samples.beta.values
        sigma_u_samples = samples.sigma_u.values
        sigma_e_samples = samples.sigma_e.values
        n_draws = len(sigma_u_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            beta_i = beta_samples[i]
            su = np.abs(sigma_u_samples[i])
            se = np.abs(sigma_e_samples[i])
            mu = X @ beta_i
            cov = su ** 2 * K_re + se ** 2 * np.eye(n)
            diff = y - mu
            try:
                L = np.linalg.cholesky(cov)
                v = np.linalg.solve(L, diff)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                pll[i] = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
