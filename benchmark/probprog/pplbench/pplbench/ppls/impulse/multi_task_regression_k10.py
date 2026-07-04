from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class MultiTaskRegressionK10(BaseImpulseImplementation):
    def __init__(self, n: int, d: int) -> None:
        self.n = n
        self.d = d

    @property
    def julia_model_path(self) -> str:
        return "standard/multi_task_regression_k10"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        data_vars = {"tau": (["draw"], samples["tau"])}
        coords = {"draw": np.arange(samples["beta1"].shape[0])}
        for k in range(1, 11):
            data_vars[f"beta{k}"] = (["draw", "feature"], samples[f"beta{k}"])
            data_vars[f"sigma{k}"] = (["draw"], samples[f"sigma{k}"])
        coords["feature"] = np.arange(samples["beta1"].shape[1])
        return xr.Dataset(data_vars, coords=coords)
