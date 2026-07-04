from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class CARSpatial(BaseNumPyroImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def model(self, data: xr.Dataset):
        Q_inv = jnp.array(data.Q_inv.values)
        log_E = jnp.array(data.log_E.values)
        y = jnp.array(data.Y.values)
        n = self.n

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
        sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(1.0))

        # Spatial effects: phi ~ MVN(0, sigma_phi^2 * Q_inv)
        Sigma_phi = sigma_phi ** 2 * Q_inv
        phi = numpyro.sample(
            "phi",
            dist.MultivariateNormal(
                loc=jnp.zeros(n), covariance_matrix=Sigma_phi
            ),
        )

        # Poisson likelihood
        log_rate = log_E + alpha + phi
        numpyro.sample("y", dist.Poisson(jnp.exp(log_rate)), obs=y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], np.asarray(samples["alpha"])),
                "sigma_phi": (["draw"], np.asarray(samples["sigma_phi"])),
                "phi": (["draw", "area"], np.asarray(samples["phi"])),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "area": np.arange(self.n),
            },
        )
