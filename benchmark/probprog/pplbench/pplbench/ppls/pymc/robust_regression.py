from typing import Dict

import numpy as np
import pymc as pm
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class RobustRegression(BasePyMCImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        alpha_scale: float,
        beta_scale: float,
        beta_loc: float,
        sigma_mean: float,
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc
        self.sigma_mean = sigma_mean

    def build_model(self, data: xr.Dataset):
        data = data.transpose("item", "feature")
        X, Y = data.X.values, data.Y.values

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=self.alpha_scale)
            beta = pm.Normal("beta", mu=self.beta_loc, sigma=self.beta_scale,
                             shape=self.k)
            # Gamma(shape=2, rate=0.1) -> PyMC: Gamma(alpha=2, beta=0.1)
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
            # Exponential(rate=1/sigma_mean) -> PyMC uses 1/lam as scale
            sigma = pm.Exponential("sigma", lam=1.0 / self.sigma_mean)
            mu = alpha + X @ beta
            pm.StudentT("Y", nu=nu, mu=mu, sigma=sigma, observed=Y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
                "nu": (["draw"], samples["nu"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
