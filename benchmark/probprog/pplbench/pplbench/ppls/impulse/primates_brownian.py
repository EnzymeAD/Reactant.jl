from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class PrimatesBrownian(BaseImpulseImplementation):
    """m14.10 as the book writes it: sicm/brownian_primates_natural.jl."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @property
    def julia_model_path(self) -> str:
        return "sicm/brownian_primates_natural"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        a = np.asarray(samples["a"]).reshape(-1)
        return xr.Dataset(
            {
                "a": (["draw"], a),
                "b": (["draw", "slope"], np.asarray(samples["b"]).reshape(len(a), 2)),
                "sigma_sq": (["draw"], np.asarray(samples["sigma_sq"]).reshape(-1)),
            },
            coords={"draw": np.arange(len(a)), "slope": np.arange(2)},
        )
