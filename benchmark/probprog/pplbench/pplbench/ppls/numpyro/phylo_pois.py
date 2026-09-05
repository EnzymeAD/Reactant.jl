# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class PhyloPois(BaseNumPyroImplementation):
    """Phylogenetic Poisson regression, written the way a NumPyro user writes it:
    the Cholesky of sigma_sq * A sits inside the model and is recomputed (and
    differentiated) at every gradient evaluation."""

    def __init__(self, n: int) -> None:
        self.n = n

    def model(self, data: xr.Dataset):
        A = jnp.array(data.A.values)
        x = jnp.array(data.X.values)
        k = jnp.array(data.Y.values)
        n = self.n

        sigma_sq = numpyro.sample("sigma_sq", dist.Exponential(1.0))
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([n]))
        b0 = numpyro.sample("b0", dist.Normal(0, 1))
        b1 = numpyro.sample("b1", dist.Normal(0, 1))
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
        eps = numpyro.sample("eps", dist.Normal(0, 1).expand([n]))

        L = jnp.linalg.cholesky(sigma_sq * A)  # chol(s * A_fixed), no jitter
        f = L @ eta
        log_rate = b0 + b1 * x + f + sigma_obs * eps
        numpyro.sample("k", dist.Poisson(rate=jnp.exp(log_rate)), obs=k)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        n_draws = len(samples["sigma_sq"])
        return xr.Dataset(
            {
                "sigma_sq": (["draw"], np.asarray(samples["sigma_sq"]).reshape(-1)),
                "eta": (["draw", "dim"], np.asarray(samples["eta"])),
                "b0": (["draw"], np.asarray(samples["b0"]).reshape(-1)),
                "b1": (["draw"], np.asarray(samples["b1"]).reshape(-1)),
                "sigma_obs": (["draw"], np.asarray(samples["sigma_obs"]).reshape(-1)),
                "eps": (["draw", "dim"], np.asarray(samples["eps"])),
            },
            coords={"draw": np.arange(n_draws), "dim": np.arange(self.n)},
        )
