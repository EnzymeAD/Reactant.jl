from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class StochasticVolatility(BaseNumPyroImplementation):
    def __init__(self, n: int, phi: float, **kwargs) -> None:
        self.T = int(kwargs.get("T", n))
        self.phi = phi

    def model(self, data: xr.Dataset):
        C_ar1 = jnp.array(data.C_ar1.values)
        y = jnp.array(data.Y.values)
        T = self.T

        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(1.0))
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([T]))

        # NCP: h = mu + sigma_h * L_C * eta
        cov_h = sigma_h ** 2 * C_ar1
        L = jnp.linalg.cholesky(cov_h)
        h = mu + L @ eta

        # y_t ~ Normal(0, exp(h_t / 2))
        numpyro.sample("y", dist.Normal(0.0, jnp.exp(h / 2.0)), obs=y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "mu": (["draw"], np.asarray(samples["mu"])),
                "sigma_h": (["draw"], np.asarray(samples["sigma_h"])),
                "eta": (["draw", "time"], np.asarray(samples["eta"])),
            },
            coords={
                "draw": np.arange(len(samples["mu"])),
                "time": np.arange(self.T),
            },
        )
