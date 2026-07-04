# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.stats import t

from .base_model import BaseModel


LOGGER = logging.getLogger(__name__)


class RobustRegression(BaseModel):
    """
    Robust Logistic Regression

    Hyper Parameters:

        n - number of items
        k - number of features
        alpha_scale, beta_scale, beta_loc -- all values in R
        sigma_mean -- values in R+

    Model:

        alpha ~ Normal(0, alpha_scale) in R
        beta ~ Normal(beta_loc, beta_scale) in R^k
        nu ~ Gamma(shape=2, scale=10.0) in R+
        sigma ~ Exponential(scale=sigma_mean) in R+

        for i in 0 .. n-1
            X_i ~ Normal(0, 10) in R^k
            mu_i = alpha + beta @ X_i
            Y_i ~ StudentT(df=nu, loc=mu, scale=sigma)

    The dataset consists of two variables X and Y and dimensions item and feature.

        X[item, feature] - float
        Y[item]          - float

    and it includes the attributes

        n
        k
        alpha_scale
        beta_scale
        beta_loc
        sigma_mean

    The posterior samples include alpha, beta, nu and sigma and dimensions draw and
    feature

        alpha[draw]         - float
        beta[draw, feature] - float
        nu[draw]            - float
        sigma[draw]         - float
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """Generate initial parameters in CONSTRAINED space."""
        rng = np.random.default_rng(seed)
        k = int(model_attrs.get("k", 10))
        return {
            "alpha": rng.uniform(-2, 2, size=(1,)).astype(np.float64),
            "beta": rng.uniform(-2, 2, size=(k,)).astype(np.float64),
            "nu": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "sigma": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2000,
        k: int = 10,
        alpha_scale: float = 10,
        beta_scale: float = 2.5,
        beta_loc: float = 0.0,
        sigma_mean: float = 10.0,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        See the class documentation for an explanation of the other parameters.
        :param train_frac: fraction of data to be used for training (default 0.5)
        """
        rng = np.random.default_rng(seed)
        alpha = rng.normal(loc=0, scale=alpha_scale)
        beta = rng.normal(loc=beta_loc, scale=beta_scale, size=k)
        nu = rng.gamma(shape=2, scale=10.0)
        sigma = rng.exponential(scale=sigma_mean)

        x = rng.normal(loc=0, scale=10, size=(n, k))
        mu = alpha + x @ beta
        y = rng.standard_t(df=nu, size=(n)) * sigma + mu

        data = xr.Dataset(
            {"X": (["item", "feature"], x), "Y": (["item"], y)},
            coords={"item": np.arange(n), "feature": np.arange(k)},
            attrs={
                "k": k,
                "alpha_scale": alpha_scale,
                "beta_scale": beta_scale,
                "beta_loc": beta_loc,
                "sigma_mean": sigma_mean,
            },
        )
        num_train = int(train_frac * n)
        train = data.isel(item=slice(None, num_train))
        test = data.isel(item=slice(num_train, None))
        train.attrs["n"] = num_train
        test.attrs["n"] = n - num_train
        return train, test

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
        nu = samples.nu.values.reshape(-1, 1)  # size = (iterations, 1)
        sigma = samples.sigma.values.reshape(-1, 1)  # size = (iterations, 1)
        x = test.X.values  # size = (k, n)
        y = test.Y.values.reshape(1, -1)  # size = (1, n)
        mu = alpha + beta @ x  # size = (iterations, n)
        logprobs = t.logpdf(x=y, loc=mu, df=nu, scale=sigma)
        return logprobs.sum(axis=1)  # size = (iterations,)
