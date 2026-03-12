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


class NSchools(BaseNumPyroImplementation):
    def __init__(
        self,
        n: int,
        num_states: int,
        num_districts_per_state: int,
        num_types: int,
        dof_baseline: float,
        scale_baseline: float,
        scale_state: float,
        scale_district: float,
        scale_type: float,
        state_idx=None,
        district_idx=None,
        type_idx=None,
    ) -> None:
        self.n = n
        self.num_states = num_states
        self.num_districts_per_state = num_districts_per_state
        self.num_types = num_types
        self.dof_baseline = dof_baseline
        self.scale_baseline = scale_baseline
        self.scale_state = scale_state
        self.scale_district = scale_district
        self.scale_type = scale_type

    def model(self, data: xr.Dataset):
        Y = data.Y.values
        sigma = data.sigma.values
        state_idx = jnp.array(data.attrs["state_idx"])
        district_idx = jnp.array(data.attrs["district_idx"])
        type_idx = jnp.array(data.attrs["type_idx"])

        sigma_state = numpyro.sample("sigma_state", dist.HalfCauchy(self.scale_state))
        sigma_district = numpyro.sample("sigma_district", dist.HalfCauchy(self.scale_district))
        sigma_type = numpyro.sample("sigma_type", dist.HalfCauchy(self.scale_type))

        beta_baseline = numpyro.sample(
            "beta_baseline", dist.StudentT(self.dof_baseline, 0.0, self.scale_baseline)
        )

        with numpyro.plate("states", self.num_states):
            beta_state = numpyro.sample("beta_state", dist.Normal(0.0, sigma_state))

        with numpyro.plate("districts_flat", self.num_states * self.num_districts_per_state):
            beta_district_flat = numpyro.sample("beta_district_flat", dist.Normal(0.0, sigma_district))

        with numpyro.plate("types", self.num_types):
            beta_type = numpyro.sample("beta_type", dist.Normal(0.0, sigma_type))

        # Compute Yhat using index lookups
        beta_district = beta_district_flat.reshape(self.num_states, self.num_districts_per_state)
        Yhat = (
            beta_baseline
            + beta_state[state_idx]
            + beta_district[state_idx, district_idx]
            + beta_type[type_idx]
        )

        with numpyro.plate("N", self.n):
            numpyro.sample("Y", dist.Normal(Yhat, jnp.array(sigma)), obs=Y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        beta_district_flat = samples["beta_district_flat"]
        num_draws = beta_district_flat.shape[0]
        beta_district = beta_district_flat.reshape(
            num_draws, self.num_states, self.num_districts_per_state
        )
        return xr.Dataset(
            {
                "sigma_state": (["draw"], samples["sigma_state"]),
                "sigma_district": (["draw"], samples["sigma_district"]),
                "sigma_type": (["draw"], samples["sigma_type"]),
                "beta_baseline": (["draw"], samples["beta_baseline"]),
                "beta_state": (["draw", "state"], samples["beta_state"]),
                "beta_district": (
                    ["draw", "state", "district"],
                    beta_district,
                ),
                "beta_type": (["draw", "type"], samples["beta_type"]),
            },
            coords={
                "draw": np.arange(num_draws),
                "state": np.arange(self.num_states),
                "district": np.arange(self.num_districts_per_state),
                "type": np.arange(self.num_types),
            },
        )
