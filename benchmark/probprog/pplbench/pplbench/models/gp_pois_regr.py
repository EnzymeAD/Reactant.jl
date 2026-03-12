# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

# posteriordb dataset: gp_pois_regr / gp_pois_regr-gp_pois_regr
_POSTERIORDB_X = np.array([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
_POSTERIORDB_K = np.array([40, 37, 29, 12, 4, 3, 9, 19, 77, 82, 33], dtype=np.float64)
_POSTERIORDB_N = 11


def _se_kernel(x, rho):
    """Squared exponential kernel matrix (no scaling, no jitter)."""
    delta = x[:, None] - x[None, :]
    return np.exp(-0.5 * (delta / rho) ** 2)


class GPPoisRegr(BaseModel):
    """
    GP Poisson Regression from posteriordb (gp_pois_regr).

    Non-centered parameterization with fixed length scale (rho) and no jitter.
    Uses the posteriordb dataset (N=11).

    Model (Stan-style):

        alpha ~ HalfNormal(2)     (positive)
        f_tilde ~ Normal(0, 1)^N  (unconstrained)

        K_base = SE(x; rho)       (fixed, sample-invariant)
        cov = alpha^2 * K_base    (no jitter)
        L = cholesky(cov)
        f = L * f_tilde

        k ~ Poisson(exp(f))       (observed)

    SICM Analysis:
        K_base is sample-invariant since rho is fixed.
        cov = alpha^2 * K_base triggers CholeskyScaleFactorizationHLO:
            chol(alpha^2 * K_base) -> alpha * chol(K_base)
        This hoists the O(N^3) Cholesky outside the MCMC loop.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """Generate initial parameters in CONSTRAINED space (positive for alpha)."""
        rng = np.random.default_rng(seed)
        n = model_attrs.get("n", _POSTERIORDB_N)
        return {
            "alpha": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "f_tilde": rng.uniform(-2, 2, size=(n,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = _POSTERIORDB_N,
        rho: float = 6.0,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Return the posteriordb gp_pois_regr dataset.

        Since this is a fixed dataset (not synthetic), we ignore `seed`.
        train = test = full dataset (GP prediction at held-out points
        requires intractable integration over the latent function).
        """
        x = _POSTERIORDB_X[:n]
        k = _POSTERIORDB_K[:n]

        data = xr.Dataset(
            {"X": (["point"], x), "Y": (["point"], k)},
            coords={"point": np.arange(n)},
            attrs={"n": n, "rho": rho},
        )

        # train = test = full dataset
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Compute Poisson log-likelihood on test data for each posterior sample.

        For each sample (alpha, f_tilde):
            K_base = SE(x; rho)
            cov = alpha^2 * K_base
            L = cholesky(cov)
            f = L @ f_tilde
            PLL = sum(k * f - exp(f) - log(k!))
        """
        x = test.X.values
        k = test.Y.values
        n = len(x)
        rho = test.attrs["rho"]

        alpha_samples = samples.alpha.values  # (n_draws, 1) or (n_draws,)
        f_tilde_samples = samples.f_tilde.values  # (n_draws, n)
        n_draws = len(alpha_samples)

        K_base = _se_kernel(x, rho)
        log_k_fact = gammaln(k + 1)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            alpha_i = np.atleast_1d(alpha_samples[i]).item()
            f_tilde_i = f_tilde_samples[i]
            cov = alpha_i ** 2 * K_base
            try:
                L = np.linalg.cholesky(cov)
                f = L @ f_tilde_i
                pll[i] = np.sum(k * f - np.exp(f) - log_k_fact)
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
