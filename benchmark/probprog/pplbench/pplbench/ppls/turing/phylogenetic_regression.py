from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class PhylogeneticRegression(BaseTuringImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @property
    def julia_model_path(self) -> str:
        return "sicm/phylogenetic_regression"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma_p": (["draw"], samples["sigma_p"]),
                "sigma_e": (["draw"], samples["sigma_e"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_p"])),
                "feature": np.arange(self.k),
            },
        )
