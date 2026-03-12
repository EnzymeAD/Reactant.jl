from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class PhylogeneticRegression(BasePyMCImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    def build_model(self, data: xr.Dataset):
        X = data.X.values
        C_phylo = data.C_phylo.values
        C_env = data.C_env.values
        y = data.Y.values

        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0.0, sigma=10.0, shape=self.k)
            sigma_p = pm.HalfNormal("sigma_p", sigma=1.0)
            sigma_e = pm.HalfNormal("sigma_e", sigma=1.0)

            mu = pt.dot(X, beta)
            cov = sigma_p ** 2 * C_phylo + sigma_e ** 2 * C_env

            pm.MvNormal("y", mu=mu, cov=cov, observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], samples["beta"]),
                "sigma_p": (["draw"], samples["sigma_p"]),
                "sigma_e": (["draw"], samples["sigma_e"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_p"])),
                "feature": np.arange(self.k),
            },
        )
