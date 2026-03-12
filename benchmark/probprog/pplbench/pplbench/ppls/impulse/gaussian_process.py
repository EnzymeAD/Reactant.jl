from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class GaussianProcess(BaseImpulseImplementation):
    def __init__(self, n: int, noise_std: float) -> None:
        self.n = n
        self.noise_std = noise_std

    @property
    def julia_model_path(self) -> str:
        return "standard/gaussian_process"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "kernel_var": (["draw"], samples["kernel_var"]),
                "kernel_length": (["draw"], samples["kernel_length"]),
                "kernel_noise": (["draw"], samples["kernel_noise"]),
            },
            coords={
                "draw": np.arange(len(samples["kernel_var"])),
            },
        )
