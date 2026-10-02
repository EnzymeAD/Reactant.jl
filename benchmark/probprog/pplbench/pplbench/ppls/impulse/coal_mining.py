from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class CoalMining(BaseImpulseImplementation):
    def __init__(
        self,
        n: int,
        s_true: int = None,
        l1_true: float = None,
        l2_true: float = None,
        **attrs,
    ) -> None:
        # The recorded series (dataset="jarrett") has no true values; its
        # attrs are n, dataset, first_year.
        self.n = n
        self.s_true = s_true
        self.l1_true = l1_true
        self.l2_true = l2_true

    @property
    def julia_model_path(self) -> str:
        return "standard/coal_mining"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        l1 = np.asarray(samples["l1"]).reshape(-1)  # (draw,)
        l2 = np.asarray(samples["l2"]).reshape(-1)  # (draw,)
        return xr.Dataset(
            {"l1": (["draw"], l1), "l2": (["draw"], l2)},
            coords={"draw": np.arange(l1.shape[0])},
        )
