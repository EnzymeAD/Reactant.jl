# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class HierarchicalMVN(BaseStanImplementation):
    def __init__(self, n: int, K: int, J: int, **kwargs) -> None:
        self.K = K
        self.J = J

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        K = int(data.attrs["K"])
        J = int(data.attrs["J"])
        return {
            "K": K,
            "J": J,
            "Omega": data.Omega.values.tolist(),
            "Y": data.Y.values.tolist(),  # K x J
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        mu = np.atleast_2d(samples["mu"])
        sigma_k = np.atleast_2d(samples["sigma_k"])
        sigma_obs = np.atleast_1d(samples["sigma_obs"])
        eta = np.atleast_2d(samples["eta"])
        if sigma_obs.ndim > 1:
            sigma_obs = sigma_obs.squeeze(1)
        return xr.Dataset(
            {
                "mu": (["draw", "outcome"], mu),
                "sigma_k": (["draw", "outcome"], sigma_k),
                "sigma_obs": (["draw"], sigma_obs),
                "eta": (["draw", "dim"], eta),
            },
            coords={
                "draw": np.arange(len(sigma_obs)),
                "outcome": np.arange(self.K),
                "dim": np.arange(self.J * self.K),
            },
        )

    def get_pars(self) -> List[str]:
        return ["mu", "sigma_k", "sigma_obs", "eta"]

    def get_code(self) -> str:
        return """
// Hierarchical MVN with NCP
// Sigma = diag(sigma_k) * Omega * diag(sigma_k)
// theta_j = mu + chol(Sigma) * eta_j
// y_{jk} ~ Normal(theta_{jk}, sigma_obs)
data {
  int<lower=1> K;
  int<lower=1> J;
  matrix[K, K] Omega;
  matrix[K, J] Y;
}
parameters {
  vector[K] mu;
  vector<lower=0>[K] sigma_k;
  real<lower=0> sigma_obs;
  vector[K * J] eta;
}
model {
  mu ~ normal(0, 10);
  sigma_k ~ normal(0, 1);  // half-normal via <lower=0>
  sigma_obs ~ normal(0, 1);
  eta ~ normal(0, 1);

  // Covariance: diag(sigma_k) * Omega * diag(sigma_k)
  matrix[K, K] Sigma = diag_matrix(sigma_k) * Omega * diag_matrix(sigma_k);
  matrix[K, K] L = cholesky_decompose(Sigma);

  // NCP transform: theta_j = mu + L * eta_j
  matrix[K, J] eta_matrix = to_matrix(eta, K, J);
  matrix[K, J] theta = rep_matrix(mu, J) + L * eta_matrix;

  // Likelihood
  for (j in 1:J) {
    Y[:, j] ~ normal(theta[:, j], sigma_obs);
  }
}
"""
