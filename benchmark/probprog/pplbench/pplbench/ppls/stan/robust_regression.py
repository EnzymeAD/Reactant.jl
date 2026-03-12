# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class RobustRegression(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")
        # we already have all the values to be bound except for X and Y in self.attrs
        attrs = self.attrs.copy()
        attrs["X"] = data.X.values
        attrs["Y"] = data.Y.values
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        # CmdStanPy may or may not include a chains dimension depending on
        # version/number of chains, so squeeze safely.
        alpha = np.atleast_1d(samples["alpha"])
        beta = np.atleast_2d(samples["beta"])
        nu = np.atleast_1d(samples["nu"])
        sigma = np.atleast_1d(samples["sigma"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        if beta.ndim > 2:
            beta = beta.squeeze(1)
        if nu.ndim > 1:
            nu = nu.squeeze(1)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "beta": (["draw", "feature"], beta),
                "nu": (["draw"], nu),
                "sigma": (["draw"], sigma),
            },
            coords={
                "draw": np.arange(beta.shape[0]),
                "feature": np.arange(beta.shape[-1]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["alpha", "beta", "nu", "sigma"]

    def get_code(self) -> str:
        """
        :returns: Returns a string the represents the Stan model.
        """
        return """
// Linear Model with Student-t Errors
data {
  // number of observations
  int n;
  // response
  vector[n] Y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  // should not include an intercept
  matrix [n, k] X;
  // priors on alpha
  real alpha_scale;
  real beta_loc;
  real beta_scale;
  real<lower=0> sigma_mean;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[k] beta;
  real<lower=0> sigma;
  // degrees of freedom;
  real<lower=0> nu;
}
transformed parameters {
  vector[n] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, alpha_scale);
  beta ~ normal(beta_loc, beta_scale);
  sigma ~ exponential(1/sigma_mean);
  // see Stan prior distribution suggestions
  nu ~ gamma(2, 0.1);
  // likelihood
  Y ~ student_t(nu, mu, sigma);
}
"""
