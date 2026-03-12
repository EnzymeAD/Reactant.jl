# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class NSchools(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # Note: self.attrs already contains the following
        # n, num_states, num_districts_per_state, num_types, dof_baseline
        # scale_baseline, scale_state, scale_district, scale_type
        attrs: dict = self.attrs.copy()
        attrs["Y"] = data.Y.values
        attrs["sigma"] = data.sigma.values
        # indices in Stan are 1-based
        # note: we are not doing in place increments because we shouldn't
        # modify the arrays that are passed in
        attrs["state_idx"] = np.array(attrs["state_idx"]) + 1
        attrs["district_idx"] = np.array(attrs["district_idx"]) + 1
        attrs["type_idx"] = np.array(attrs["type_idx"]) + 1
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        def _squeeze(arr):
            return arr.squeeze() if arr.ndim > 1 else arr

        num_draws = samples["beta_baseline"].shape[0]
        beta_district = _squeeze(samples["beta_district"]).reshape(
            num_draws, self.attrs["num_states"], self.attrs["num_districts_per_state"]
        )
        return xr.Dataset(
            {
                "sigma_state": (["draw"], _squeeze(samples["sigma_state"])),
                "sigma_district": (["draw"], _squeeze(samples["sigma_district"])),
                "sigma_type": (["draw"], _squeeze(samples["sigma_type"])),
                "beta_baseline": (["draw"], _squeeze(samples["beta_baseline"])),
                "beta_state": (["draw", "state"], _squeeze(samples["beta_state"])),
                "beta_district": (
                    ["draw", "state", "district"],
                    beta_district,
                ),
                "beta_type": (["draw", "type"], _squeeze(samples["beta_type"])),
            },
            coords={
                "draw": np.arange(num_draws),
                "state": np.arange(self.attrs["num_states"]),
                "district": np.arange(self.attrs["num_districts_per_state"]),
                "type": np.arange(self.attrs["num_types"]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return [
            "sigma_state",
            "sigma_district",
            "sigma_type",
            "beta_baseline",
            "beta_state",
            "beta_district",
            "beta_type",
        ]

    def get_code(self) -> str:
        """
        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
  int<lower=1> n; // number of schools
  int<lower=1> num_states;
  int<lower=1> num_districts_per_state;
  int<lower=1> num_types;
  array[n] real Y;
  array[n] real<lower=0> sigma;
  array[n] int<lower=1, upper=num_states> state_idx;
  array[n] int<lower=1, upper=num_districts_per_state> district_idx;
  array[n] int<lower=1, upper=num_types> type_idx;
  real<lower=0> dof_baseline;
  real<lower=0> scale_baseline;
  real<lower=0> scale_state;
  real<lower=0> scale_district;
  real<lower=0> scale_type;
}

parameters {
  real<lower=0> sigma_state;
  real<lower=0> sigma_district;
  real<lower=0> sigma_type;
  real beta_baseline;
  vector[num_states] beta_state;
  matrix[num_states, num_districts_per_state] beta_district;
  vector[num_types] beta_type;
}

transformed parameters {
    vector[n] Yhat;
    for (i in 1:n) {
        Yhat[i] = beta_baseline
                + beta_state[state_idx[i]]
                + beta_district[state_idx[i], district_idx[i]]
                + beta_type[type_idx[i]];
    }
}

model {
  sigma_state ~ cauchy(0, scale_state);
  sigma_district ~ cauchy(0, scale_district);
  sigma_type ~ cauchy(0, scale_type);
  beta_baseline ~ student_t(dof_baseline, 0, scale_baseline);
  beta_state ~ normal(0, sigma_state);
  to_vector(beta_district) ~ normal(0, sigma_district);
  beta_type ~ normal(0, sigma_type);
  Y ~ normal(Yhat, sigma);
}
"""
