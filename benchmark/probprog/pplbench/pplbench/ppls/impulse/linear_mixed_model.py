from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class LinearMixedModel(BaseImpulseImplementation):
    def __init__(self, n: int, k: int, q: int) -> None:
        self.n = n
        self.k = k
        self.q = q

    @property
    def julia_model_path(self) -> str:
        return "sicm/linear_mixed_model"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma_u": (["draw"], samples["sigma_u"]),
                "sigma_e": (["draw"], samples["sigma_e"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_u"])),
                "feature": np.arange(self.k),
            },
        )
