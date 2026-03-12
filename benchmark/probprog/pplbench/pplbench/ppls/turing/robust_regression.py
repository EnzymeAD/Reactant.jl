from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class RobustRegression(BaseTuringImplementation):
    def __init__(self, n: int, k: int, alpha_scale: float, beta_scale: float,
                 beta_loc: float, sigma_mean: float) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc
        self.sigma_mean = sigma_mean

    @property
    def julia_model_path(self) -> str:
        return "standard/robust_regression"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
                "nu": (["draw"], samples["nu"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
