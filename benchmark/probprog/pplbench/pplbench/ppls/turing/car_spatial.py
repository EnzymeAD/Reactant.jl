from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class CARSpatial(BaseTuringImplementation):
    def __init__(self, n: int, rho: float, **kwargs) -> None:
        self.n = n
        self.rho = rho

    @property
    def julia_model_path(self) -> str:
        return "sicm/car_spatial"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "sigma_phi": (["draw"], samples["sigma_phi"]),
                "phi": (["draw", "dim"], samples["phi"]),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "dim": np.arange(self.n),
            },
        )
