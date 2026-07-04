from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class ScaleFamilyMVN(BaseImpulseImplementation):
    def __init__(self, n: int, lengthscale: float) -> None:
        self.n = n
        self.lengthscale = lengthscale

    @property
    def julia_model_path(self) -> str:
        return "sicm/scale_family_mvn"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {"tau": (["draw"], samples["tau"])},
            coords={"draw": np.arange(len(samples["tau"]))},
        )
