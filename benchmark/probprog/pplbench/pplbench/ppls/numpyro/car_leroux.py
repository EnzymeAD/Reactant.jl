# NumPyro implementation of the Leroux CAR model with rho estimated
# (models/car_leroux.py). Input: the model's xarray data; output: the draws of
# alpha, sigma_phi, rho and phi as an xarray Dataset.
from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class CARLeroux(BaseNumPyroImplementation):
    """Leroux CAR, rho estimated (models/car_leroux.py). The spatial field is
    stated with its precision matrix, tau (rho R + (1 - rho) I), R = D - W."""

    def __init__(self, n: int) -> None:
        self.n = n

    def model(self, data: xr.Dataset):
        R = jnp.array(data.R.values)
        log_E = jnp.array(data.log_E.values)
        y = jnp.array(data.Y.values)
        n = self.n

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
        sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 1.0))

        tau = 1.0 / sigma_phi ** 2
        P = tau * (rho * R + (1.0 - rho) * jnp.eye(n))
        phi = numpyro.sample(
            "phi",
            dist.MultivariateNormal(loc=jnp.zeros(n), precision_matrix=P),
        )

        numpyro.sample("y", dist.Poisson(jnp.exp(log_E + alpha + phi)), obs=y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], np.asarray(samples["alpha"])),
                "sigma_phi": (["draw"], np.asarray(samples["sigma_phi"])),
                "rho": (["draw"], np.asarray(samples["rho"])),
                "phi": (["draw", "area"], np.asarray(samples["phi"])),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "area": np.arange(self.n),
            },
        )


class CARLerouxEigh(CARLeroux):
    """The same target density with the log-determinant of the precision taken
    from a once-computed eigendecomposition of R (R = U diag(lam) U'), so
    log det P = sum(log(tau (rho lam + 1 - rho))). Mathematically identical to
    CARLeroux; used only as the NumPyro reference for the n = 1921 map, where
    the O(n^3) per-gradient factorization of CARLeroux makes a 400-iteration
    reference run impractically long. Equality of the two log densities is
    checked pointwise at n = 56 in the validation script."""

    def model(self, data: xr.Dataset):
        R = jnp.array(data.R.values)
        lam = jnp.array(np.linalg.eigvalsh(np.asarray(data.R.values)))
        log_E = jnp.array(data.log_E.values)
        y = jnp.array(data.Y.values)
        n = self.n

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
        sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.Uniform(0.0, 1.0))
        tau = 1.0 / sigma_phi ** 2

        phi = numpyro.sample("phi", dist.ImproperUniform(
            dist.constraints.real_vector, (), (n,)))
        quad = tau * (rho * phi @ (R @ phi) + (1.0 - rho) * phi @ phi)
        logdet = jnp.sum(jnp.log(tau * (rho * lam + 1.0 - rho)))
        numpyro.factor("phi_car", 0.5 * logdet - 0.5 * quad
                       - 0.5 * n * jnp.log(2 * jnp.pi))

        numpyro.sample("y", dist.Poisson(jnp.exp(log_E + alpha + phi)), obs=y)
