from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class LoopSharedChol(BaseImpulseImplementation):
    def __init__(self, n: int, d: int, T: int) -> None:
        self.n = n
        self.d = d
        self.T = T

    @property
    def julia_model_path(self) -> str:
        return "standard/loop_shared_chol"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {"sigma": (["draw"], samples["sigma"])},
            coords={"draw": np.arange(len(samples["sigma"]))},
        )
