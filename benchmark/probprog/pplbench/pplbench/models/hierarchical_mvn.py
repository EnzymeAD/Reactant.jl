# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _random_correlation_matrix(rng, K):
    """Generate a random correlation matrix via the vine method."""
    # Start with a random lower triangular matrix
    A = rng.standard_normal((K, K))
    Q, _ = np.linalg.qr(A)
    eigenvalues = rng.uniform(0.5, 2.0, K)
    S = Q @ np.diag(eigenvalues) @ Q.T
    # Convert to correlation
    D_inv = np.diag(1.0 / np.sqrt(np.diag(S)))
    R = D_inv @ S @ D_inv
    # Ensure exact symmetry and unit diagonal
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


class HierarchicalMVN(BaseModel):
    """
    Hierarchical MVN: theta_j = mu + L * eta_j, y_{jk} ~ Normal(theta_{jk}, sigma_obs)
    Sigma = diag(sigma_k) * Omega * diag(sigma_k)

    SICM: DiagonalScaleCholeskyFactorization (NEW) on diag(sigma_k) * Omega * diag(sigma_k).
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        K = int(model_attrs.get("K", 3))
        J = int(model_attrs.get("J", 20))
        return {
            "mu": rng.uniform(-2, 2, size=(K,)).astype(np.float64),
            "sigma_k": np.exp(rng.uniform(-1, 1, size=(K,))).astype(np.float64),
            "sigma_obs": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
            "eta": rng.standard_normal(J * K).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 60,
        K: int = 3,
        J: int = 20,
        true_sigma_obs: float = 0.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Known outcome correlation matrix
        Omega = _random_correlation_matrix(rng, K)
        Omega += 1e-6 * np.eye(K)

        # True per-outcome scales
        true_sigma_k = np.exp(rng.uniform(-0.5, 1.0, K))

        # True grand means
        true_mu = rng.uniform(-2, 2, K)

        # Build covariance: diag(sigma_k) * Omega * diag(sigma_k)
        D = np.diag(true_sigma_k)
        Sigma = D @ Omega @ D
        L = np.linalg.cholesky(Sigma)

        # Generate group effects and observations
        Y = np.zeros((K, J))
        for j in range(J):
            eta_j = rng.standard_normal(K)
            theta_j = true_mu + L @ eta_j
            Y[:, j] = theta_j + rng.standard_normal(K) * true_sigma_obs

        data = xr.Dataset(
            {
                "Omega": (["outcome_i", "outcome_j"], Omega),
                "Y": (["outcome", "group"], Y),
            },
            coords={
                "outcome_i": np.arange(K),
                "outcome_j": np.arange(K),
                "outcome": np.arange(K),
                "group": np.arange(J),
            },
            attrs={"K": K, "J": J, "n": K * J},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        Y = test.Y.values  # K x J
        Omega = test.Omega.values
        K, J = Y.shape

        mu_samples = samples.mu.values          # n_draws x K
        sigma_k_samples = samples.sigma_k.values  # n_draws x K
        sigma_obs_samples = samples.sigma_obs.values  # n_draws
        eta_samples = samples.eta.values        # n_draws x (J*K)
        n_draws = len(sigma_obs_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            mu_i = mu_samples[i]
            sk_i = np.abs(sigma_k_samples[i])
            so_i = np.abs(sigma_obs_samples[i])
            eta_i = eta_samples[i].reshape(K, J)

            D = np.diag(sk_i)
            Sigma = D @ Omega @ D
            try:
                L = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                pll[i] = -1e10
                continue

            ll = 0.0
            for j in range(J):
                theta_j = mu_i + L @ eta_i[:, j]
                diff = Y[:, j] - theta_j
                ll += np.sum(
                    -0.5 * np.log(2 * np.pi) - np.log(so_i) - 0.5 * (diff / so_i) ** 2
                )
            pll[i] = ll

        return pll
