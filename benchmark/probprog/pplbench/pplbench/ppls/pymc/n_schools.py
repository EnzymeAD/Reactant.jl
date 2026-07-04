from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class NSchools(BasePyMCImplementation):
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

    def build_model(self, data: xr.Dataset):
        Y = data.Y.values
        sigma = data.sigma.values
        state_idx = np.array(data.attrs["state_idx"])
        district_idx = np.array(data.attrs["district_idx"])
        type_idx = np.array(data.attrs["type_idx"])

        num_districts_total = self.num_states * self.num_districts_per_state

        with pm.Model() as model:
            sigma_state = pm.HalfCauchy("sigma_state", beta=self.scale_state)
            sigma_district = pm.HalfCauchy("sigma_district", beta=self.scale_district)
            sigma_type = pm.HalfCauchy("sigma_type", beta=self.scale_type)

            beta_baseline = pm.StudentT(
                "beta_baseline", nu=self.dof_baseline, mu=0.0,
                sigma=self.scale_baseline
            )

            beta_state = pm.Normal(
                "beta_state", mu=0.0, sigma=sigma_state,
                shape=self.num_states
            )
            beta_district_flat = pm.Normal(
                "beta_district_flat", mu=0.0, sigma=sigma_district,
                shape=num_districts_total
            )
            beta_type = pm.Normal(
                "beta_type", mu=0.0, sigma=sigma_type,
                shape=self.num_types
            )

            # Vectorized index lookup
            beta_district = beta_district_flat.reshape(
                (self.num_states, self.num_districts_per_state)
            )
            Yhat = (
                beta_baseline
                + beta_state[state_idx]
                + beta_district[state_idx, district_idx]
                + beta_type[type_idx]
            )

            pm.Normal("Y", mu=Yhat, sigma=sigma, observed=Y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
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
