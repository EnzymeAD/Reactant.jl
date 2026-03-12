# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.linalg import cho_factor, cho_solve

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _kernel_se(X, Z, var, length, noise, jitter=1e-6):
    """Squared exponential kernel: var * exp(-0.5 * ||x-z||^2 / length^2) + noise*I"""
    delta = X[:, None] - Z[None, :]
    K = var * np.exp(-0.5 * (delta / length) ** 2)
    if X is Z or (X.shape == Z.shape and np.allclose(X, Z)):
        K += (noise + jitter) * np.eye(X.shape[0])
    return K


class GaussianProcess(BaseModel):
    """
    Gaussian Process Regression with Squared Exponential Kernel

    Hyper Parameters:

        n - number of data points
        noise_std - observation noise standard deviation for data generation

    Model:

        kernel_var    ~ LogNormal(0, 10)   (positive)
        kernel_length ~ LogNormal(0, 10)   (positive)
        kernel_noise  ~ LogNormal(0, 10)   (positive)

        Y ~ MultivariateNormal(0, K(X, X))

    where K is the squared exponential kernel:
        K(x, z) = kernel_var * exp(-0.5 * ||x - z||^2 / kernel_length^2)
                + kernel_noise * I

    The dataset consists of:
        X[point] - float, input locations
        Y[point] - float, observations

    and includes the attributes: n, noise_std

    The posterior samples include:
        kernel_var[draw]    - float
        kernel_length[draw] - float
        kernel_noise[draw]  - float

    SICM Analysis:
        The pairwise distance matrix X[:, None] - X is sample-invariant.
        Only the scaling by kernel_length and kernel_var is sample-dependent.
        With SICM, the O(N^2) pairwise computation is hoisted outside the
        MCMC loop and computed once.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """Generate initial parameters in CONSTRAINED space (positive for kernel params)."""
        rng = np.random.default_rng(seed)
        return {
            "kernel_var": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "kernel_length": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "kernel_noise": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 60,
        noise_std: float = 0.1,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Generate synthetic GP regression data.

        True function: f(x) = sin(2*pi*x) + 0.5*cos(4*pi*x)
        with additive Gaussian noise. Y is standardized to zero mean, unit variance.
        """
        rng = np.random.default_rng(seed)

        X = np.linspace(0, 1, n)
        f_true = np.sin(2 * np.pi * X) + 0.5 * np.cos(4 * np.pi * X)
        Y = f_true + noise_std * rng.standard_normal(n)

        # Standardize Y
        Y = (Y - Y.mean()) / Y.std()

        data = xr.Dataset(
            {"X": (["point"], X), "Y": (["point"], Y)},
            coords={"point": np.arange(n)},
            attrs={"n": n, "noise_std": noise_std},
        )

        # Random split for spatial coverage
        indices = np.arange(n)
        rng2 = np.random.default_rng(seed + 1)
        rng2.shuffle(indices)
        n_train = int(n * train_frac)
        train_idx = np.sort(indices[:n_train])
        test_idx = np.sort(indices[n_train:])

        train = data.isel(point=train_idx)
        train.attrs["n"] = len(train_idx)
        test = data.isel(point=test_idx)
        test.attrs["n"] = len(test_idx)

        return train, test

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Compute GP marginal log-likelihood on test data for each posterior sample.

        For each sample (kernel_var, kernel_length, kernel_noise):
            K_test = K(X_test, X_test; theta)
            PLL = log N(Y_test | 0, K_test)
        """
        X_test = test.X.values
        Y_test = test.Y.values
        n_test = len(X_test)

        kernel_var = samples.kernel_var.values
        kernel_length = samples.kernel_length.values
        kernel_noise = samples.kernel_noise.values
        n_draws = len(kernel_var)

        pll = np.zeros(n_draws)

        for i in range(n_draws):
            K = _kernel_se(X_test, X_test, kernel_var[i], kernel_length[i], kernel_noise[i])
            try:
                L, low = cho_factor(K, lower=True)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                alpha = cho_solve((L, low), Y_test)
                pll[i] = -0.5 * (
                    Y_test @ alpha + log_det + n_test * np.log(2 * np.pi)
                )
            except np.linalg.LinAlgError:
                pll[i] = -1e10

        return pll
