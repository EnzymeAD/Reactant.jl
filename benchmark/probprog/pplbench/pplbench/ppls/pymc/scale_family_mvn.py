from typing import Dict

import numpy as np
import pymc as pm
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class ScaleFamilyMVN(BasePyMCImplementation):
    def __init__(self, n: int, lengthscale: float) -> None:
        self.n = n
        self.lengthscale = lengthscale

    def build_model(self, data: xr.Dataset):
        R = data.R.values
        y = data.Y.values

        with pm.Model() as model:
            tau = pm.HalfNormal("tau", sigma=1.0)
            cov = tau ** 2 * R
            pm.MvNormal("y", mu=np.zeros(self.n), cov=cov, observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {"tau": (["draw"], samples["tau"])},
            coords={"draw": np.arange(len(samples["tau"]))},
        )
