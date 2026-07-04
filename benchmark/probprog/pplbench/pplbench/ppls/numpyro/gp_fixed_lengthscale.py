from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class GPFixedLengthscale(BaseNumPyroImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def model(self, data: xr.Dataset):
        x = jnp.array(data.X.values)
        y = jnp.array(data.Y.values)

        alpha = numpyro.sample("alpha", dist.HalfNormal(1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # SE kernel with fixed rho (sample-invariant)
        delta = x[:, None] - x[None, :]
        K_base = jnp.exp(-0.5 * (delta / self.rho) ** 2)

        cov = alpha ** 2 * K_base + sigma ** 2 * jnp.eye(self.n)
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
            {
                "alpha": (["draw"], np.asarray(samples["alpha"])),
                "sigma": (["draw"], np.asarray(samples["sigma"])),
            },
            coords={"draw": np.arange(len(samples["alpha"]))},
        )
