from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class PhyloPois(BaseImpulseImplementation):
    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def julia_model_path(self) -> str:
        return "motivating/phylo_pois"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        n_draws = len(samples["sigma_sq"])
        return xr.Dataset(
            {
                "sigma_sq": (["draw"], np.asarray(samples["sigma_sq"]).reshape(-1)),
                "eta": (["draw", "dim"], np.asarray(samples["eta"])),
                "b0": (["draw"], np.asarray(samples["b0"]).reshape(-1)),
                "b1": (["draw"], np.asarray(samples["b1"]).reshape(-1)),
                "sigma_obs": (["draw"], np.asarray(samples["sigma_obs"]).reshape(-1)),
                "eps": (["draw", "dim"], np.asarray(samples["eps"])),
            },
            coords={"draw": np.arange(n_draws), "dim": np.arange(self.n)},
        )
