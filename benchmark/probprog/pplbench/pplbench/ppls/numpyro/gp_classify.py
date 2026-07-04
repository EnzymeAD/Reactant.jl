from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


class GPClassify(BaseNumPyroImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def model(self, data: xr.Dataset):
        x = jnp.array(data.X.values)
        y = jnp.array(data.Y.values)
        n = self.n

        alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
        f_tilde = numpyro.sample("f_tilde", dist.Normal(0, 1).expand([n]))

        # SE kernel with fixed rho
        delta = x[:, None] - x[None, :]
        K_base = jnp.exp(-0.5 * (delta / self.rho) ** 2)

        cov = alpha ** 2 * K_base
        L = jnp.linalg.cholesky(cov)
        f = L @ f_tilde

        numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], np.asarray(samples["alpha"])),
                "f_tilde": (["draw", "dim"], np.asarray(samples["f_tilde"])),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "dim": np.arange(self.n),
            },
        )
