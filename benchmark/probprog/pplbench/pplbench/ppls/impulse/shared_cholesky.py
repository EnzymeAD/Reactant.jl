from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class SharedCholesky(BaseImpulseImplementation):
    def __init__(self, n: int, **kwargs) -> None:
        self.n = n

    @property
    def julia_model_path(self) -> str:
        return "ablation/shared_cholesky"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        phi = np.array(samples["phi"])
        sigma = np.array(samples["sigma"])
        return xr.Dataset(
            {
                "phi": (["draw"], phi.ravel()),
                "sigma": (["draw"], sigma.ravel()),
            },
            coords={"draw": np.arange(len(phi.ravel()))},
        )
