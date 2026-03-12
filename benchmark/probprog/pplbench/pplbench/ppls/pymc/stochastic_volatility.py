from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class StochasticVolatility(BasePyMCImplementation):
    def __init__(self, n: int, phi: float, **kwargs) -> None:
        self.T = int(kwargs.get("T", n))
        self.phi = phi

    def build_model(self, data: xr.Dataset):
        C_ar1 = data.C_ar1.values
        y = data.Y.values

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0.0, sigma=5.0)
            sigma_h = pm.HalfNormal("sigma_h", sigma=1.0)
            eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=self.T)

            # NCP: h = mu + chol(sigma_h^2 * C_ar1) @ eta
            cov_h = sigma_h ** 2 * C_ar1
            L = pt.linalg.cholesky(cov_h)
            h = mu + pt.dot(L, eta)

            pm.Normal("y", mu=0.0, sigma=pt.exp(h / 2.0), observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw"], samples["mu"]),
                "sigma_h": (["draw"], samples["sigma_h"]),
                "eta": (["draw", "time"], samples["eta"]),
            },
            coords={
                "draw": np.arange(len(samples["mu"])),
                "time": np.arange(self.T),
            },
        )
