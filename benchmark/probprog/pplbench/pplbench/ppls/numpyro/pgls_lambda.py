# NumPyro implementation of PGLS with Pagel's lambda estimated
# (models/pgls_lambda.py). Input: the model's xarray data (C, X, Y); output: the
# draws of beta, sigma and lambda as an xarray Dataset.
from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class PGLSLambda(BaseNumPyroImplementation):
    """PGLS with Pagel's lambda estimated (models/pgls_lambda.py)."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    def model(self, data: xr.Dataset):
        C = jnp.array(data.C.values)
        X = jnp.array(data.X.values)
        y = jnp.array(data.Y.values)
        n, k = self.n, self.k

        beta = numpyro.sample("beta", dist.Normal(0.0, 10.0).expand([k]).to_event(1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        lam = numpyro.sample("lambda", dist.Uniform(0.0, 1.0))

        Sigma = sigma ** 2 * (lam * C + (1.0 - lam) * jnp.eye(n))
        numpyro.sample(
            "y", dist.MultivariateNormal(loc=X @ beta, covariance_matrix=Sigma),
            obs=y,
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], np.asarray(samples["beta"])),
                "sigma": (["draw"], np.asarray(samples["sigma"])),
                "lambda": (["draw"], np.asarray(samples["lambda"])),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
                "feature": np.arange(self.k),
            },
        )
