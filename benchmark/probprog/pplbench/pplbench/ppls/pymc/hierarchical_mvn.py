from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class HierarchicalMVN(BasePyMCImplementation):
    def __init__(self, n: int, K: int, J: int, **kwargs) -> None:
        self.K = K
        self.J = J

    def build_model(self, data: xr.Dataset):
        Omega = data.Omega.values  # K x K
        Y = data.Y.values  # K x J

        K = self.K
        J = self.J

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0.0, sigma=10.0, shape=K)
            sigma_k = pm.HalfNormal("sigma_k", sigma=1.0, shape=K)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
            eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=J * K)

            # Covariance: diag(sigma_k) * Omega * diag(sigma_k)
            cov = pt.outer(sigma_k, sigma_k) * Omega
            L = pt.linalg.cholesky(cov)

            # NCP: theta_j = mu + L * eta_j
            eta_matrix = eta.reshape((K, J))
            theta = mu[:, None] + pt.dot(L, eta_matrix)  # K x J

            # Observations
            pm.Normal("y", mu=theta.ravel(), sigma=sigma_obs,
                      observed=Y.ravel())

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw", "outcome"], samples["mu"]),
                "sigma_k": (["draw", "outcome"], samples["sigma_k"]),
                "sigma_obs": (["draw"], samples["sigma_obs"]),
                "eta": (["draw", "dim"], samples["eta"]),
            },
            coords={
                "draw": np.arange(len(samples["sigma_obs"])),
                "outcome": np.arange(self.K),
                "dim": np.arange(self.J * self.K),
            },
        )
