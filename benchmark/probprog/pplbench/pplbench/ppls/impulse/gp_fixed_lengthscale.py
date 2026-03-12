from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class GPFixedLengthscale(BaseImpulseImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    @property
    def julia_model_path(self) -> str:
        return "sicm/gp_fixed_lengthscale"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={"draw": np.arange(len(samples["alpha"]))},
        )
