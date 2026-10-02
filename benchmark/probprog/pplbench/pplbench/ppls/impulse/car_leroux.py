# Impulse implementation of the Leroux CAR model with rho estimated
# (models/car_leroux.py). Runs the Julia model sicm/car_leroux.jl through
# BaseImpulseImplementation and returns the draws of alpha, sigma_phi, rho and
# phi as an xarray Dataset.
from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class CARLeroux(BaseImpulseImplementation):
    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def julia_model_path(self) -> str:
        return "sicm/car_leroux"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], np.asarray(samples["alpha"]).reshape(-1)),
                "sigma_phi": (["draw"], np.asarray(samples["sigma_phi"]).reshape(-1)),
                "rho": (["draw"], np.asarray(samples["rho"]).reshape(-1)),
                "phi": (["draw", "area"], np.asarray(samples["phi"])),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "area": np.arange(self.n),
            },
        )
