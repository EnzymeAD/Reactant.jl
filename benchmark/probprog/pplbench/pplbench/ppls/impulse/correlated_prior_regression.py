from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class CorrelatedPriorRegression(BaseImpulseImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @property
    def julia_model_path(self) -> str:
        return "standard/correlated_prior_regression"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
