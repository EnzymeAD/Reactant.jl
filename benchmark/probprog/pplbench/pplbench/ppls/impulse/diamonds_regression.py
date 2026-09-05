from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class DiamondsRegression(BaseImpulseImplementation):
    def __init__(self, n: int, k: int, kc: int) -> None:
        self.n = n
        self.k = k
        self.kc = kc

    @property
    def julia_model_path(self) -> str:
        return "sicm/diamonds_regression"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
                "feature": np.arange(self.k),
            },
        )
