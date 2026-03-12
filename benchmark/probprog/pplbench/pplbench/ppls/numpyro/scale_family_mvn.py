from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class ScaleFamilyMVN(BaseNumPyroImplementation):
    def __init__(self, n: int, lengthscale: float) -> None:
        self.n = n
        self.lengthscale = lengthscale

    def model(self, data: xr.Dataset):
        R = jnp.array(data.R.values)
        y = jnp.array(data.Y.values)

        tau = numpyro.sample("tau", dist.HalfNormal(1.0))

        cov = tau ** 2 * R
        numpyro.sample(
            "y",
            dist.MultivariateNormal(
                loc=jnp.zeros(self.n), covariance_matrix=cov
            ),
            obs=y,
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {"tau": (["draw"], np.asarray(samples["tau"]))},
            coords={"draw": np.arange(len(samples["tau"]))},
        )
