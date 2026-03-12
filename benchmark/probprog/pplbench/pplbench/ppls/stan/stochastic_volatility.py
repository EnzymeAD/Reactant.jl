# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class StochasticVolatility(BaseStanImplementation):
    def __init__(self, n: int, phi: float, **kwargs) -> None:
        self.T = int(kwargs.get("T", n))

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "T": int(data.attrs["T"]),
            "C_ar1": data.C_ar1.values.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        mu = np.atleast_1d(samples["mu"])
        sigma_h = np.atleast_1d(samples["sigma_h"])
        eta = np.atleast_2d(samples["eta"])
        if mu.ndim > 1:
            mu = mu.squeeze(1)
        if sigma_h.ndim > 1:
            sigma_h = sigma_h.squeeze(1)
        return xr.Dataset(
            {
                "mu": (["draw"], mu),
                "sigma_h": (["draw"], sigma_h),
                "eta": (["draw", "time"], eta),
            },
            coords={
                "draw": np.arange(len(mu)),
                "time": np.arange(self.T),
            },
        )

    def get_pars(self) -> List[str]:
        return ["mu", "sigma_h", "eta"]

    def get_code(self) -> str:
        return """
// Stochastic Volatility with NCP
// h = mu + chol(sigma_h^2 * C_ar1) * eta, y_t ~ Normal(0, exp(h_t / 2))
data {
  int<lower=1> T;
  matrix[T, T] C_ar1;
  vector[T] Y;
}
parameters {
  real mu;
  real<lower=0> sigma_h;
  vector[T] eta;
}
model {
  mu ~ normal(0, 5);
  sigma_h ~ normal(0, 1);  // half-normal via <lower=0>
  eta ~ normal(0, 1);

  matrix[T, T] cov_h = sigma_h^2 * C_ar1;
  matrix[T, T] L = cholesky_decompose(cov_h);
  vector[T] h = mu + L * eta;

  Y ~ normal(0, exp(h / 2));
}
"""
