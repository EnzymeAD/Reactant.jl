from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class PhylogeneticRegression(BaseNumPyroImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    def model(self, data: xr.Dataset):
        X = jnp.array(data.X.values)
        C_phylo = jnp.array(data.C_phylo.values)
        C_env = jnp.array(data.C_env.values)
        y = jnp.array(data.Y.values)

        with numpyro.plate("features", self.k):
            beta = numpyro.sample("beta", dist.Normal(0.0, 10.0))

        sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(1.0))
        sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(1.0))

        mu = X @ beta
        cov = sigma_p ** 2 * C_phylo + sigma_e ** 2 * C_env

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
                "sigma_p": (["draw"], np.asarray(samples["sigma_p"])),
                "sigma_e": (["draw"], np.asarray(samples["sigma_e"])),
            },
            coords={
                "draw": np.arange(len(samples["sigma_p"])),
                "feature": np.arange(self.k),
            },
        )
