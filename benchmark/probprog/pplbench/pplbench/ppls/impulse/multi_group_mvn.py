from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class MultiGroupMVN(BaseImpulseImplementation):
    def __init__(self, n: int, K: int) -> None:
        self.n = n
        self.K = K

    @property
    def julia_model_path(self) -> str:
        return "sicm/multi_group_mvn"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
            },
        )
