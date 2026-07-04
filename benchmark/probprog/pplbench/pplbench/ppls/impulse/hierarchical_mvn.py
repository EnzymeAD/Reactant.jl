from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class HierarchicalMVN(BaseImpulseImplementation):
    def __init__(self, n: int, K: int, J: int, **kwargs) -> None:
        self.K = K
        self.J = J

    @property
    def julia_model_path(self) -> str:
        return "sicm/hierarchical_mvn"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw", "outcome"], samples["mu"]),
                "sigma_k": (["draw", "outcome"], samples["sigma_k"]),
                "sigma_obs": (["draw"], samples["sigma_obs"]),
                "eta": (["draw", "dim"], samples["eta"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_obs"])),
                "outcome": np.arange(self.K),
                "dim": np.arange(self.J * self.K),
            },
        )
