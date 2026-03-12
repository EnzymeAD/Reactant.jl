from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class NSchools(BaseTuringImplementation):
    def __init__(self, n: int, num_states: int, num_districts_per_state: int,
                 num_types: int, dof_baseline: float, scale_baseline: float,
                 scale_state: float, scale_district: float, scale_type: float,
                 state_idx=None, district_idx=None, type_idx=None) -> None:
        self.n = n
        self.num_states = num_states
        self.num_districts_per_state = num_districts_per_state
        self.num_types = num_types

    @property
    def julia_model_path(self) -> str:
        return "standard/n_schools"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
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
