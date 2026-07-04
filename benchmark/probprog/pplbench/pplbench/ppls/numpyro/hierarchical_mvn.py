from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class HierarchicalMVN(BaseNumPyroImplementation):
    def __init__(self, n: int, K: int, J: int, **kwargs) -> None:
        self.K = K
        self.J = J

    def model(self, data: xr.Dataset):
        Omega = jnp.array(data.Omega.values)
        Y = jnp.array(data.Y.values)  # K x J
        K = self.K
        J = self.J

        with numpyro.plate("outcomes", K):
            mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
            sigma_k = numpyro.sample("sigma_k", dist.HalfNormal(1.0))

        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([J * K]))

        # Covariance: diag(sigma_k) * Omega * diag(sigma_k)
        cov = jnp.outer(sigma_k, sigma_k) * Omega
        L = jnp.linalg.cholesky(cov)

        # NCP: theta_j = mu + L * eta_j
        eta_matrix = eta.reshape(K, J)
        theta = mu[:, None] + L @ eta_matrix  # K x J

        # Observations
        numpyro.sample(
            "y",
            dist.Normal(theta.ravel(), sigma_obs),
            obs=Y.ravel(),
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw", "outcome"], np.asarray(samples["mu"])),
                "sigma_k": (["draw", "outcome"], np.asarray(samples["sigma_k"])),
                "sigma_obs": (["draw"], np.asarray(samples["sigma_obs"])),
                "eta": (["draw", "dim"], np.asarray(samples["eta"])),
            },
            coords={
                "draw": np.arange(len(samples["sigma_obs"])),
                "outcome": np.arange(self.K),
                "dim": np.arange(self.J * self.K),
            },
        )
