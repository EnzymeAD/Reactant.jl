# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class LinearMixedModel(BaseStanImplementation):
    def __init__(self, n: int, k: int, q: int, **kwargs) -> None:
        self.n = n
        self.k = k

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "n": int(data.attrs["n"]),
            "k": int(data.attrs["k"]),
            "X": data.X.values.tolist(),
            "K_re": data.K_re.values.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        beta = np.atleast_2d(samples["beta"])
        sigma_u = np.atleast_1d(samples["sigma_u"])
        sigma_e = np.atleast_1d(samples["sigma_e"])
        if sigma_u.ndim > 1:
            sigma_u = sigma_u.squeeze(1)
        if sigma_e.ndim > 1:
            sigma_e = sigma_e.squeeze(1)
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], beta),
                "sigma_u": (["draw"], sigma_u),
                "sigma_e": (["draw"], sigma_e),
            },
            coords={
                "draw": np.arange(len(sigma_u)),
                "feature": np.arange(self.k),
            },
        )

    def get_pars(self) -> List[str]:
        return ["beta", "sigma_u", "sigma_e"]

    def get_code(self) -> str:
        return """
// Linear Mixed Model
// y ~ MVN(X * beta, sigma_u^2 * K_re + sigma_e^2 * I)
data {
  int<lower=1> n;
  int<lower=1> k;
  matrix[n, k] X;
  matrix[n, n] K_re;
  vector[n] Y;
}
parameters {
  vector[k] beta;
  real<lower=0> sigma_u;
  real<lower=0> sigma_e;
}
model {
  beta ~ normal(0, 1);
  sigma_u ~ normal(0, 1);  // half-normal via <lower=0>
  sigma_e ~ normal(0, 1);

  matrix[n, n] cov = sigma_u^2 * K_re + sigma_e^2 * diag_matrix(rep_vector(1, n));
  Y ~ multi_normal(X * beta, cov);
}
"""
