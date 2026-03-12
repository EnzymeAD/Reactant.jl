# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


def _kernel_se(X, Z, var, length, noise, jitter=1.0e-6):
    """Squared exponential kernel (matching Reactant implementation)."""
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


class GaussianProcess(BaseNumPyroImplementation):
    def __init__(self, n: int, noise_std: float) -> None:
        self.n = n
        self.noise_std = noise_std

    def model(self, data: xr.Dataset):
        X = jnp.array(data.X.values)
        Y = jnp.array(data.Y.values)

        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))
        noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))

        k = _kernel_se(X, X, var, length, noise)
        numpyro.sample(
            "Y",
            dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
            obs=Y,
        )

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "kernel_var": (["draw"], np.asarray(samples["kernel_var"])),
                "kernel_length": (["draw"], np.asarray(samples["kernel_length"])),
                "kernel_noise": (["draw"], np.asarray(samples["kernel_noise"])),
            },
            coords={
                "draw": np.arange(len(samples["kernel_var"])),
            },
        )
