import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel
from .utils import split_train_test


LOGGER = logging.getLogger(__name__)


class CorrelatedPriorRegression(BaseModel):
    """
    Bayesian Linear Regression with a dense correlated prior on coefficients.

    Model:
        beta ~ MultivariateNormal(0, Sigma_prior)   in R^k
        sigma ~ HalfNormal(1)                        in R+
        y_i ~ Normal(X_i @ beta, sigma)              for i = 1..n

    Sigma_prior is a dense (non-diagonal) k x k positive definite matrix,
    constructed as A @ A' + I where A is random. This makes the Cholesky
    decomposition of Sigma_prior an O(k^3) operation that is
    sample-invariant (hoistable by hoist-enzyme-regions).

    The dataset consists of X (n x k design matrix), Y (n responses),
    and Sigma_prior (k x k covariance matrix).
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 30))
        return {
            "beta": rng.uniform(-2, 2, size=(k,)).astype(np.float64),
            "sigma": np.array([1.0]),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 200,
        k: int = 30,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Build a dense positive definite prior covariance
        A = rng.standard_normal((k, k)) * 0.5
        Sigma_prior = A @ A.T + np.eye(k)

        # True parameters from the prior
        L_true = np.linalg.cholesky(Sigma_prior)
        beta_true = L_true @ rng.standard_normal(k)
        sigma_true = 1.0

        # Design matrix and responses
        X = rng.standard_normal((n, k))
        mu = X @ beta_true
        y = mu + rng.standard_normal(n) * sigma_true

        data = xr.Dataset(
            {
                "X": (["item", "feature"], X),
                "Y": (["item"], y),
                "Sigma_prior": (["feature_i", "feature_j"], Sigma_prior),
            },
            coords={
                "item": np.arange(n),
                "feature": np.arange(k),
                "feature_i": np.arange(k),
                "feature_j": np.arange(k),
            },
            attrs={"n": n, "k": k},
        )
        return split_train_test(data, "item", train_frac)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        samples = samples.transpose("draw", "feature")
        test = test.transpose("feature", "item", ...)
        beta = samples.beta.values               # (draws, k)
        sigma = np.abs(samples.sigma.values)      # (draws,)
        x = test.X.values                         # (k, n_test)
        y = test.Y.values.reshape(1, -1)          # (1, n_test)

        mu = beta @ x                             # (draws, n_test)
        residual = y - mu                          # (draws, n_test)
        loglike = (
            -0.5 * (residual / sigma.reshape(-1, 1)) ** 2
            - np.log(sigma.reshape(-1, 1))
            - 0.5 * np.log(2 * np.pi)
        )
        return loglike.sum(axis=1)
