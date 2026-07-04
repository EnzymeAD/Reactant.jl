from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class CARSpatial(BasePyMCImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def build_model(self, data: xr.Dataset):
        Q_inv = data.Q_inv.values
        log_E = data.log_E.values
        y = data.Y.values

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
            sigma_phi = pm.HalfNormal("sigma_phi", sigma=1.0)

            # Spatial effects: phi ~ MVN(0, sigma_phi^2 * Q_inv)
            Sigma_phi = sigma_phi ** 2 * Q_inv
            phi = pm.MvNormal("phi", mu=np.zeros(self.n), cov=Sigma_phi)

            # Poisson likelihood
            log_rate = log_E + alpha + phi
            pm.Poisson("y", mu=pt.exp(log_rate), observed=y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "sigma_phi": (["draw"], samples["sigma_phi"]),
                "phi": (["draw", "area"], samples["phi"]),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "area": np.arange(self.n),
            },
        )
