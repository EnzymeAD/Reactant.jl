from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class GPFixedLengthscale(BasePyMCImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def build_model(self, data: xr.Dataset):
        x = data.X.values
        y = data.Y.values

        delta = x[:, None] - x[None, :]
        K_base = np.exp(-0.5 * (delta / self.rho) ** 2)

        with pm.Model() as model:
            alpha = pm.HalfNormal("alpha", sigma=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            cov = alpha ** 2 * K_base + sigma ** 2 * np.eye(self.n)
            pm.MvNormal("y", mu=np.zeros(self.n), cov=cov, observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={"draw": np.arange(len(samples["alpha"]))},
        )
