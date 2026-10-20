from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class PrimatesBrownian(BaseTuringImplementation):
    """m14.10 as the book writes it: turing/motivating/primates_brownian.jl."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @property
    def julia_model_path(self) -> str:
        return "motivating/primates_brownian"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        a = np.asarray(samples["a"], dtype=np.float64).reshape(-1)
        return xr.Dataset(
            {
                "a": (["draw"], a),
                "b": (["draw", "slope"], np.stack(
                    [np.asarray(samples["bM"], dtype=np.float64).reshape(-1),
                     np.asarray(samples["bG"], dtype=np.float64).reshape(-1)], axis=1)),
                "sigma_sq": (["draw"], np.asarray(samples["sigma_sq"], dtype=np.float64).reshape(-1)),
            },
            coords={"draw": np.arange(len(a)), "slope": np.arange(2)},
        )
