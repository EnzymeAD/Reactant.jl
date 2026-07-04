# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class PhylogeneticRegression(BaseStanImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "n": int(data.attrs["n"]),
            "k": int(data.attrs["k"]),
            "X": data.X.values.tolist(),
            "C_phylo": data.C_phylo.values.tolist(),
            "C_env": data.C_env.values.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        beta = np.atleast_2d(samples["beta"])
        sigma_p = np.atleast_1d(samples["sigma_p"])
        sigma_e = np.atleast_1d(samples["sigma_e"])
        if sigma_p.ndim > 1:
            sigma_p = sigma_p.squeeze(1)
        if sigma_e.ndim > 1:
            sigma_e = sigma_e.squeeze(1)
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], beta),
                "sigma_p": (["draw"], sigma_p),
                "sigma_e": (["draw"], sigma_e),
            },
            coords={
                "draw": np.arange(len(sigma_p)),
                "feature": np.arange(self.k),
            },
        )

    def get_pars(self) -> List[str]:
        return ["beta", "sigma_p", "sigma_e"]

    def get_code(self) -> str:
        return """
// Phylogenetic Regression
// y ~ MVN(X * beta, sigma_p^2 * C_phylo + sigma_e^2 * C_env)
data {
  int<lower=1> n;
  int<lower=1> k;
  matrix[n, k] X;
  matrix[n, n] C_phylo;
  matrix[n, n] C_env;
  vector[n] Y;
}
parameters {
  vector[k] beta;
  real<lower=0> sigma_p;
  real<lower=0> sigma_e;
}
model {
  beta ~ normal(0, 10);
  sigma_p ~ normal(0, 1);  // half-normal via <lower=0>
  sigma_e ~ normal(0, 1);

  matrix[n, n] K = sigma_p^2 * C_phylo + sigma_e^2 * C_env;
  Y ~ multi_normal(X * beta, K);
}
"""
