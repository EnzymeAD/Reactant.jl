# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class RobustRegression(BaseNumPyroImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        alpha_scale: float,
        beta_scale: float,
        beta_loc: float,
        sigma_mean: float,
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc
        self.sigma_mean = sigma_mean

    def matched_init_params(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """Deterministic canonical init (matched-init control experiment).

        Constrained point identical to the Impulse arm's NUTS start
        (standard/robust_regression.jl `build_constraint`): alpha = 0, beta = 0
        (real support -> z = 0), nu = 1 and sigma = 1 (positive support with an
        exp transform -> z = log(1) = 0). Routed through init_to_value by
        numpyro_harness.py; only used with --matched-init.
        """
        return {
            "alpha": np.zeros(1),
            "beta": np.zeros(self.k),
            "nu": np.ones(1),
            "sigma": np.ones(1),
        }

    def model(self, data: xr.Dataset):
        data = data.transpose("item", "feature")
        X, Y = data.X.values, data.Y.values

        alpha = numpyro.sample("alpha", dist.Normal(scale=self.alpha_scale))
        with numpyro.plate("K", self.k):
            beta = numpyro.sample("beta", dist.Normal(self.beta_loc, self.beta_scale))
        nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
        sigma = numpyro.sample("sigma", dist.Exponential(1 / self.sigma_mean))
        mu = alpha + X @ beta
        with numpyro.plate("N", self.n):
            numpyro.sample("Y", dist.StudentT(nu, mu, sigma), obs=Y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
                "nu": (["draw"], samples["nu"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
