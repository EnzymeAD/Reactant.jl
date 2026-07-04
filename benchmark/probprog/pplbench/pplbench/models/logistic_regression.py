# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel
from .utils import log1pexp, split_train_test


LOGGER = logging.getLogger(__name__)


class LogisticRegression(BaseModel):
    """
    Bayesian Logistic Regression

    Hyper Parameters:

        n - number of items
        k - number of features
        alpha_scale, beta_scale, beta_loc, rho -- all values in R

    Model:

        alpha ~ Normal(0, alpha_scale) in R
        beta ~ Normal(beta_loc, beta_scale) in R^k
        x_scales ~ exp(normal(0, rho)) in R^k

        for i in 0 .. n-1
            X_i ~ normal(0, x_scales) in R^k
            mu_i = alpha + beta @ X_i
            # p_i = probability that X_i belongs to class 1
            p_i = sigmoid(mu_i)
            Y_i ~ Bernoulli(p_i)  for i = 1..n

    The dataset consists of two variables X and Y and dimensions item and feature.

        X[item, feature] - float
        Y[item]          - integer 0 or 1

    and it includes the attributes

        n
        k
        alpha_scale
        beta_scale
        beta_loc

    The posterior samples include alpha and beta and dimensions draw and feature

        alpha[draw]         - float
        beta[draw, feature] - float
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """
        Sample initial parameters from Uniform(-2, 2), matching NumPyro's
        default ``init_to_uniform(radius=2)`` strategy.
        """
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 10))
        return {
            "alpha": rng.uniform(-2, 2, size=(1,)).astype(np.float64),
            "beta": rng.uniform(-2, 2, size=(k,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2000,
        k: int = 10,
        alpha_scale: float = 10,
        beta_scale: float = 2.5,
        beta_loc: float = 0.0,
        rho: float = 3,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        See the class documentation for an explanation of the other parameters.
        :param train_frac: fraction of data to be used for training (default 0.5)
        """
        rng = np.random.default_rng(seed)
        alpha = rng.normal(loc=0, scale=alpha_scale)
        beta = rng.normal(loc=beta_loc, scale=beta_scale, size=k)
        x_scales = rng.lognormal(mean=0, sigma=rho, size=(1, k))
        x = rng.normal(loc=0, scale=x_scales, size=(n, k))
        mu = alpha + x @ beta
        # to avoid overflow or underflow we will only evaluate the sigmoid when |mu| < 20
        prob = 1.0 / (1.0 + np.exp(-mu, where=np.abs(mu) < 20))
        prob[mu >= 20] = 1.0
        prob[mu <= -20] = 0.0
        if sum(np.abs(mu) >= 20) == n:
            LOGGER.warn("All logits are too extreme! Consider reducing 'rho'")
        y = rng.binomial(n=1, p=prob).astype(dtype=np.int32).reshape(-1)

        data = xr.Dataset(
            {"X": (["item", "feature"], x), "Y": (["item"], y)},
            coords={"item": np.arange(n), "feature": np.arange(k)},
            attrs={
                "n": n,
                "k": k,
                "alpha_scale": alpha_scale,
                "beta_scale": beta_scale,
                "beta_loc": beta_loc,
            },
        )
        return split_train_test(data, "item", train_frac)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Computes the predictive likelihood of all the test items w.r.t. each sample.
        See the class documentation for the `samples` and `test` parameters.
        :returns: a numpy array of the same size as the sample dimension.
        """
        # transpose the datasets to be in a convenient format
        samples = samples.transpose("draw", "feature")
        test = test.transpose("feature", "item")
        alpha = samples.alpha.values.reshape(-1, 1)  # size = (iterations, 1)
        beta = samples.beta.values  # size = (iterations, k)
        x = test.X.values  # size = (k, n_test)
        y = test.Y.values.reshape(1, -1)  # size = (1, n_test)
        mu = alpha + beta @ x  # size = (iterations, n_test)
        # for y=1 the log_prob is -log(1 + exp(-mu))
        # and for y=0 the log_prob is -log(1 + exp(mu))
        sign_y = np.where(y, -1, 1)  # size = (1, n_test)
        loglike = -log1pexp(sign_y * mu)  # size = (iterations, n_test)
        return loglike.sum(axis=1)  # size = (iterations,)
