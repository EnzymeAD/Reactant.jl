# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class CARSpatial(BaseStanImplementation):
    def __init__(self, n: int, rho: float, **kwargs) -> None:
        self.n = n

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "n": int(data.attrs["n"]),
            "Q_inv": data.Q_inv.values.tolist(),
            "log_E": data.log_E.values.tolist(),
            "Y": data.Y.values.astype(int).tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        alpha = np.atleast_1d(samples["alpha"])
        sigma_phi = np.atleast_1d(samples["sigma_phi"])
        phi = np.atleast_2d(samples["phi"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        if sigma_phi.ndim > 1:
            sigma_phi = sigma_phi.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "sigma_phi": (["draw"], sigma_phi),
                "phi": (["draw", "area"], phi),
            },
            coords={
                "draw": np.arange(len(alpha)),
                "area": np.arange(self.n),
            },
        )

    def get_pars(self) -> List[str]:
        return ["alpha", "sigma_phi", "phi"]

    def get_code(self) -> str:
        return """
// CAR Spatial Model
// phi ~ MVN(0, sigma_phi^2 * Q_inv), y ~ Poisson(exp(log_E + alpha + phi))
data {
  int<lower=1> n;
  matrix[n, n] Q_inv;
  vector[n] log_E;
  array[n] int Y;
}
parameters {
  real alpha;
  real<lower=0> sigma_phi;
  vector[n] phi;
}
model {
  alpha ~ normal(0, 10);
  sigma_phi ~ normal(0, 1);  // half-normal via <lower=0>
  phi ~ multi_normal(rep_vector(0, n), sigma_phi^2 * Q_inv);

  Y ~ poisson_log(log_E + alpha + phi);
}
"""
