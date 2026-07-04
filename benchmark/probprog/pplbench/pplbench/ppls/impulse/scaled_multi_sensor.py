from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class ScaledMultiSensor(BaseImpulseImplementation):
    def __init__(self, n: int, d_state: int) -> None:
        self.n = n
        self.d_state = d_state

    @property
    def julia_model_path(self) -> str:
        return "sicm/scaled_multi_sensor"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "sigma": (["draw"], samples["sigma"]),
                "f_tilde": (["draw", "dim"], samples["f_tilde"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
                "dim": np.arange(self.n),
            },
        )
