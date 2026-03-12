from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class LinearMixedModel(BaseNumPyroImplementation):
    def __init__(self, n: int, k: int, q: int) -> None:
        self.n = n
        self.k = k
        self.q = q

    def model(self, data: xr.Dataset):
        X = jnp.array(data.X.values)
        K_re = jnp.array(data.K_re.values)
        y = jnp.array(data.Y.values)

        with numpyro.plate("features", self.k):
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(1.0))
        sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(1.0))

        mu = X @ beta
        cov = sigma_u ** 2 * K_re + sigma_e ** 2 * jnp.eye(self.n)

        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=mu, covariance_matrix=cov),
            obs=y,
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], np.asarray(samples["beta"])),
                "sigma_u": (["draw"], np.asarray(samples["sigma_u"])),
                "sigma_e": (["draw"], np.asarray(samples["sigma_e"])),
            },
            coords={
                "draw": np.arange(len(samples["sigma_u"])),
                "feature": np.arange(self.k),
            },
        )
