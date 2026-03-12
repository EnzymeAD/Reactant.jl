from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class StochasticVolatility(BaseTuringImplementation):
    def __init__(self, n: int, phi: float, **kwargs) -> None:
        self.n = n
        self.phi = phi

    @property
    def julia_model_path(self) -> str:
        return "sicm/stochastic_volatility"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw"], samples["mu"]),
                "sigma_h": (["draw"], samples["sigma_h"]),
                "eta": (["draw", "dim"], samples["eta"]),
            },
            coords={
                "draw": np.arange(len(samples["mu"])),
                "dim": np.arange(self.n),
            },
        )
