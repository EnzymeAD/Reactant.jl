from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class SARLag(BaseImpulseImplementation):
    def __init__(self, n: int, k: int, rho_lo: float, rho_hi: float) -> None:
        self.n = n
        self.k = k
        self.rho_lo = rho_lo
        self.rho_hi = rho_hi

    @property
    def julia_model_path(self) -> str:
        return "sicm/sar_lag_diagmul"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "rho_raw": (["draw"], np.asarray(samples["rho_raw"]).reshape(-1)),
                "beta": (["draw", "feature"], np.asarray(samples["beta"])),
                "sigma": (["draw"], np.asarray(samples["sigma"]).reshape(-1)),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
                "feature": np.arange(self.k),
            },
        )
