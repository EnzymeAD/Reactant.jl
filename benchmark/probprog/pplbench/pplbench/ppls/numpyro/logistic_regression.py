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

from .base_numpyro_impl import MATCHED_INIT_REAL, BaseNumPyroImplementation


class LogisticRegression(BaseNumPyroImplementation):
    def __init__(
        self, n: int, k: int, alpha_scale: float, beta_scale: float, beta_loc: float
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc

    def matched_init_params(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """Deterministic canonical init (matched-init control experiment).

        Constrained point identical to the Impulse arm's NUTS start
        (standard/logistic_regression.jl `build_constraint`): alpha and beta both
        at MATCHED_INIT_REAL. Both sites have real support, so under NumPyro's
        identity transform this is also the unconstrained point. Routed through
        init_to_value by numpyro_harness.py; only used with --matched-init.

        NOT the origin: `mu = alpha + X @ beta` is linear in these two sites, so
        alpha = beta = 0 would put all n logits at EXACTLY 0, which is the one
        point where numpyro's Bernoulli(logits=...) reports a wrong gradient (see
        MATCHED_INIT_REAL in base_numpyro_impl.py). The offset must stay in step
        with the Impulse arm's `build_constraint`, or the gate stops comparing
        the same starting point.
        """
        return {
            "alpha": np.full(1, MATCHED_INIT_REAL),
            "beta": np.full(self.k, MATCHED_INIT_REAL),
        }

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
