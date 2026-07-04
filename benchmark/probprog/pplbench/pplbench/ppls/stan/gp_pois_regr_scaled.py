# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class GPPoisRegrScaled(BaseStanImplementation):
    def __init__(self, n: int, rho: float, **kwargs) -> None:
        self.n = n
        self.rho = rho

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        X = data.X.values
        rho = float(data.attrs["rho"])
        n = int(data.attrs["n"])
        delta = X[:, None] - X[None, :]
        K_base = np.exp(-0.5 * (delta / rho) ** 2)
        return {
            "n": n,
            "K_base": K_base.tolist(),
            "Y": data.Y.values.astype(int).tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        alpha = np.atleast_1d(samples["alpha"])
        f_tilde = np.atleast_2d(samples["f_tilde"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "f_tilde": (["draw", "dim"], f_tilde),
            },
            coords={
                "draw": np.arange(len(alpha)),
                "dim": np.arange(self.n),
            },
        )

    def get_pars(self) -> List[str]:
        return ["alpha", "f_tilde"]

    def get_code(self) -> str:
        return """
// GP Poisson Regression (Scaled) with NCP
// f = chol(alpha^2 * K_base) * f_tilde, k ~ Poisson(exp(f))
data {
  int<lower=1> n;
  matrix[n, n] K_base;
  array[n] int Y;
}
parameters {
  real<lower=0> alpha;
  vector[n] f_tilde;
}
model {
  alpha ~ normal(0, 2);  // half-normal via <lower=0>
  f_tilde ~ normal(0, 1);

  matrix[n, n] cov = alpha^2 * K_base;
  matrix[n, n] L = cholesky_decompose(cov);
  vector[n] f = L * f_tilde;

  Y ~ poisson_log(f);
}
"""
