from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class LinearMixedModel(BasePyMCImplementation):
    def __init__(self, n: int, k: int, q: int) -> None:
        self.n = n
        self.k = k
        self.q = q

    def build_model(self, data: xr.Dataset):
        X = data.X.values
        K_re = data.K_re.values
        y = data.Y.values

        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=self.k)
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_e = pm.HalfNormal("sigma_e", sigma=1.0)

            mu = pt.dot(X, beta)
            cov = sigma_u ** 2 * K_re + sigma_e ** 2 * np.eye(self.n)

            pm.MvNormal("y", mu=mu, cov=cov, observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma_u": (["draw"], samples["sigma_u"]),
                "sigma_e": (["draw"], samples["sigma_e"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_u"])),
                "feature": np.arange(self.k),
            },
        )
