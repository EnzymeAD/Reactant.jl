# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _random_pd_matrix(rng, n, condition_number=10.0):
    """Generate a random positive definite matrix with bounded condition number."""
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    eigenvalues = np.linspace(1.0, condition_number, n)
    return Q @ np.diag(eigenvalues) @ Q.T


class PhylogeneticRegression(BaseModel):
    """
    Phylogenetic Regression: y ~ MVN(X @ beta, sigma_p^2 * C_phylo + sigma_e^2 * C_env)

    Two invariant covariance sources: phylogenetic tree + environmental similarity.

    SICM: GeneralizedEigenLift (NEW) on sigma_p^2 * C_phylo + sigma_e^2 * C_env.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 3))
        return {
            "beta": rng.uniform(-2, 2, size=(k,)).astype(np.float64),
            "sigma_p": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
            "sigma_e": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 50,
        k: int = 3,
        true_sigma_p: float = 1.5,
        true_sigma_e: float = 1.0,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Fixed-effects design matrix
        X = rng.standard_normal((n, k))

        # Phylogenetic covariance (from a tree: shared branch lengths)
        C_phylo = _random_pd_matrix(rng, n, condition_number=20.0)
        # Normalize so trace ~ n
        C_phylo *= n / np.trace(C_phylo)

        # Environmental similarity (from habitat data)
        C_env = _random_pd_matrix(rng, n, condition_number=10.0)
        C_env *= n / np.trace(C_env)

        # True parameters
        beta_true = rng.standard_normal(k)

        # Generate data
        mu = X @ beta_true
        cov = true_sigma_p ** 2 * C_phylo + true_sigma_e ** 2 * C_env
        L = np.linalg.cholesky(cov)
        y = mu + L @ rng.standard_normal(n)

        data = xr.Dataset(
            {
                "X": (["species", "feature"], X),
                "C_phylo": (["i", "j"], C_phylo),
                "C_env": (["i", "j"], C_env),
                "Y": (["species"], y),
            },
            coords={
                "species": np.arange(n),
                "feature": np.arange(k),
                "i": np.arange(n),
                "j": np.arange(n),
            },
            attrs={"n": n, "k": k},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        X = test.X.values
        C_phylo = test.C_phylo.values
        C_env = test.C_env.values
        y = test.Y.values
        n = len(y)

        beta_samples = samples.beta.values
        sigma_p_samples = samples.sigma_p.values
        sigma_e_samples = samples.sigma_e.values
        n_draws = len(sigma_p_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            beta_i = beta_samples[i]
            sp = np.abs(sigma_p_samples[i])
            se = np.abs(sigma_e_samples[i])
            mu = X @ beta_i
            cov = sp ** 2 * C_phylo + se ** 2 * C_env
            diff = y - mu
            try:
                L = np.linalg.cholesky(cov)
                v = np.linalg.solve(L, diff)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                pll[i] = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
