# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class GaussianProcess(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        self.attrs = attrs

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        # Only include variables declared in the Stan data block
        return {
            "n": int(data.X.shape[0]),
            "X": data.X.values.tolist(),
            "Y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        kernel_var = np.atleast_1d(samples["kernel_var"])
        kernel_length = np.atleast_1d(samples["kernel_length"])
        kernel_noise = np.atleast_1d(samples["kernel_noise"])
        if kernel_var.ndim > 1:
            kernel_var = kernel_var.squeeze(1)
        if kernel_length.ndim > 1:
            kernel_length = kernel_length.squeeze(1)
        if kernel_noise.ndim > 1:
            kernel_noise = kernel_noise.squeeze(1)
        return xr.Dataset(
            {
                "kernel_var": (["draw"], kernel_var),
                "kernel_length": (["draw"], kernel_length),
                "kernel_noise": (["draw"], kernel_noise),
            },
            coords={
                "draw": np.arange(len(kernel_var)),
            },
        )

    def get_pars(self) -> List[str]:
        return ["kernel_var", "kernel_length", "kernel_noise"]

    def get_code(self) -> str:
        return """
// Gaussian Process Regression with Squared Exponential Kernel
data {
  int<lower=1> n;
  array[n] real X;
  vector[n] Y;
}
parameters {
  real<lower=0> kernel_var;
  real<lower=0> kernel_length;
  real<lower=0> kernel_noise;
}
model {
  matrix[n, n] K = gp_exp_quad_cov(X, sqrt(kernel_var), kernel_length);
  for (i in 1:n) {
    K[i, i] = K[i, i] + kernel_noise + 1e-6;
  }

  kernel_var ~ lognormal(0, 10);
  kernel_length ~ lognormal(0, 10);
  kernel_noise ~ lognormal(0, 10);

  Y ~ multi_normal(rep_vector(0, n), K);
}
"""
