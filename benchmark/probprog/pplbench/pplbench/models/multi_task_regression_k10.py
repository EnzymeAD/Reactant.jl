import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel
from .utils import split_train_test

LOGGER = logging.getLogger(__name__)
K = 10


class MultiTaskRegressionK10(BaseModel):
    """Multi-Task Regression with K=10 shared-covariance tasks."""

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        d = int(model_attrs.get("d", 20))
        params = {"tau": np.array([1.0])}
        for k in range(1, K + 1):
            params[f"beta{k}"] = rng.uniform(-2, 2, size=(d,)).astype(np.float64)
            params[f"sigma{k}"] = np.array([1.0])
        return params

    @staticmethod
    def generate_data(
        seed: int, n: int = 100, d: int = 20, train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((d, d)) * 0.3
        Sigma_prior = A @ A.T + np.eye(d)
        X = rng.standard_normal((n, d))
        L = np.linalg.cholesky(Sigma_prior)

        data_vars = {
            "X": (["item", "feature"], X),
            "Sigma_prior": (["feature_i", "feature_j"], Sigma_prior),
        }
        coords = {
            "item": np.arange(n), "feature": np.arange(d),
            "feature_i": np.arange(d), "feature_j": np.arange(d),
        }
        for k in range(1, K + 1):
            beta_k = L @ rng.standard_normal(d)
            y_k = X @ beta_k + rng.standard_normal(n)
            data_vars[f"Y{k}"] = (["item"], y_k)

        data = xr.Dataset(data_vars, coords=coords, attrs={"n": n, "d": d})
        return split_train_test(data, "item", train_frac)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset,
    ) -> np.ndarray:
        test = test.transpose("feature", "item", ...)
        x = test.X.values
        total_loglike = 0.0
        for k in range(1, K + 1):
            beta = samples[f"beta{k}"].values
            sigma = np.abs(samples[f"sigma{k}"].values)
            y = test[f"Y{k}"].values.reshape(1, -1)
            mu = beta @ x
            residual = y - mu
            loglike = (
                -0.5 * (residual / sigma.reshape(-1, 1)) ** 2
                - np.log(sigma.reshape(-1, 1))
                - 0.5 * np.log(2 * np.pi)
            )
            total_loglike = total_loglike + loglike.sum(axis=1)
        return total_loglike
