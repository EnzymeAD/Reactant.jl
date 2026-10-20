from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class PrimatesBrownian(BaseNumPyroImplementation):
    """m14.10 as the book writes it (McElreath 2020, R code 14.51): the covariance
    R * sigma_sq goes to the built-in MultivariateNormal, which factors it at every
    evaluation."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    def model(self, data: xr.Dataset):
        M = jnp.array(data.X.values[:, 1])
        G = jnp.array(data.X.values[:, 2])
        R = jnp.array(data.R.values)
        B = jnp.array(data.Y.values)

        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        bM = numpyro.sample("bM", dist.Normal(0.0, 0.5))
        bG = numpyro.sample("bG", dist.Normal(0.0, 0.5))
        sigma_sq = numpyro.sample("sigma_sq", dist.Exponential(1.0))

        mu = a + bM * M + bG * G
        SIGMA = R * sigma_sq

        numpyro.sample(
            "B",
            dist.MultivariateNormal(loc=mu, covariance_matrix=SIGMA),
            obs=B,
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        a = np.asarray(samples["a"]).reshape(-1)
        return xr.Dataset(
            {
                "a": (["draw"], a),
                "b": (["draw", "slope"], np.stack(
                    [np.asarray(samples["bM"]).reshape(-1),
                     np.asarray(samples["bG"]).reshape(-1)], axis=1)),
                "sigma_sq": (["draw"], np.asarray(samples["sigma_sq"]).reshape(-1)),
            },
            coords={"draw": np.arange(len(a)), "slope": np.arange(2)},
        )
