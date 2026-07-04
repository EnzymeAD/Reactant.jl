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


class LogisticRegression(BaseNumPyroImplementation):
    def __init__(
        self, n: int, k: int, alpha_scale: float, beta_scale: float, beta_loc: float
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc

    def model(self, data: xr.Dataset):
        data = data.transpose("item", "feature")
        X, Y = data.X.values, data.Y.values
        alpha = numpyro.sample("alpha", dist.Normal(scale=self.alpha_scale))
        with numpyro.plate("K", self.k):
            beta = numpyro.sample("beta", dist.Normal(self.beta_loc, self.beta_scale))
        mu = alpha + X @ beta
        with numpyro.plate("N", self.n):
            numpyro.sample("Y", dist.Bernoulli(logits=mu), obs=Y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
