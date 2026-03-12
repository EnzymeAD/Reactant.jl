from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class StochasticVolatility(BaseImpulseImplementation):
    def __init__(self, n: int, phi: float, **kwargs) -> None:
        self.T = int(kwargs.get("T", n))
        self.phi = phi

    @property
    def julia_model_path(self) -> str:
        return "sicm/stochastic_volatility"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw"], samples["mu"]),
                "sigma_h": (["draw"], samples["sigma_h"]),
                "eta": (["draw", "time"], samples["eta"]),
            },
            coords={
                "draw": np.arange(len(samples["mu"])),
                "time": np.arange(self.T),
            },
        )
