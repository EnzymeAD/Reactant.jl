from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class EigenLiftSolve(BaseImpulseImplementation):
    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def julia_model_path(self) -> str:
        return "sicm/eigenlift_solve"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "sigma": (["draw"], samples["sigma"]),
                "f_tilde": (["draw", "dim"], samples["f_tilde"]),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "dim": np.arange(self.n),
            },
        )
