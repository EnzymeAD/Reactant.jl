# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class ScaleFamilyMVN(BaseStanImplementation):
    def __init__(self, n: int, lengthscale: float, **kwargs) -> None:
        self.n = n

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "n": int(data.attrs["n"]),
            "R": data.R.values.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        tau = np.atleast_1d(samples["tau"])
        if tau.ndim > 1:
            tau = tau.squeeze(1)
        return xr.Dataset(
            {"tau": (["draw"], tau)},
            coords={"draw": np.arange(len(tau))},
        )

    def get_pars(self) -> List[str]:
        return ["tau"]

    def get_code(self) -> str:
        return """
// Scale-Family MVN
// y ~ MVN(0, tau^2 * R)
data {
  int<lower=1> n;
  matrix[n, n] R;
  vector[n] Y;
}
parameters {
  real<lower=0> tau;
}
model {
  tau ~ normal(0, 1);  // half-normal via <lower=0>
  Y ~ multi_normal(rep_vector(0, n), tau^2 * R);
}
"""
