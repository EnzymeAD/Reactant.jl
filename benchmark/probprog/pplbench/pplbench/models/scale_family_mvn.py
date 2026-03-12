# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _exponential_correlation(locs, lengthscale):
    """Build spatial exponential correlation matrix from locations."""
    diff = locs[:, None, :] - locs[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return np.exp(-dist / lengthscale)


class ScaleFamilyMVN(BaseModel):
    """
    Scale-Family MVN: y ~ MVN(0, tau^2 * R)

    SICM cascade (O(N^3) -> O(1) per iteration):
      - CholeskyScaleFactorization
      - TriangularSolveScaleFactorization
      - LogMultiplyDistribution
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "tau": np.exp(rng.uniform(-1, 1, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 60,
        lengthscale: float = 3.0,
        true_tau: float = 2.0,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Generate spatial locations on a grid or random
        locs = rng.uniform(0, 10, size=(n, 1))

        # Build correlation matrix
        R = _exponential_correlation(locs, lengthscale)
        # Ensure positive definiteness
        R += 1e-6 * np.eye(n)

        # Generate data from the true model
        cov = true_tau ** 2 * R
        L = np.linalg.cholesky(cov)
        y = L @ rng.standard_normal(n)

        data = xr.Dataset(
            {
                "R": (["i", "j"], R),
                "Y": (["point"], y),
            },
            coords={
                "i": np.arange(n),
                "j": np.arange(n),
                "point": np.arange(n),
            },
            attrs={"n": n, "lengthscale": lengthscale},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        R = test.R.values
        y = test.Y.values
        n = len(y)

        tau_samples = samples.tau.values
        n_draws = len(tau_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            tau_i = np.abs(tau_samples[i])
            cov = tau_i ** 2 * R
            try:
                L = np.linalg.cholesky(cov)
                v = np.linalg.solve(L, y)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                pll[i] = -0.5 * (np.dot(v, v) + log_det + n * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
