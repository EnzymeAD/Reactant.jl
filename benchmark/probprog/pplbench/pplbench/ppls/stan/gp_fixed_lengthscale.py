# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class GPFixedLengthscale(BaseStanImplementation):
    def __init__(self, n: int, rho: float, **kwargs) -> None:
        self.n = n

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        X = data.X.values
        rho = float(data.attrs["rho"])
        n = int(data.attrs["n"])
        delta = X[:, None] - X[None, :]
        K_base = np.exp(-0.5 * (delta / rho) ** 2)
        return {
            "n": n,
            "K_base": K_base.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        alpha = np.atleast_1d(samples["alpha"])
        sigma = np.atleast_1d(samples["sigma"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "sigma": (["draw"], sigma),
            },
            coords={"draw": np.arange(len(alpha))},
        )

    def get_pars(self) -> List[str]:
        return ["alpha", "sigma"]

    def get_code(self) -> str:
        return """
// GP with Fixed Lengthscale
// y ~ MVN(0, alpha^2 * K_base + sigma^2 * I)
data {
  int<lower=1> n;
  matrix[n, n] K_base;
  vector[n] Y;
}
parameters {
  real<lower=0> alpha;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 1);  // half-normal via <lower=0>
  sigma ~ normal(0, 1);

  matrix[n, n] cov = alpha^2 * K_base + sigma^2 * diag_matrix(rep_vector(1, n));
  Y ~ multi_normal(rep_vector(0, n), cov);
}
"""
