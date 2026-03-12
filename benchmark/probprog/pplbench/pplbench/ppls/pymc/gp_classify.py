from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class GPClassify(BasePyMCImplementation):
    def __init__(self, n: int, rho: float) -> None:
        self.n = n
        self.rho = rho

    def build_model(self, data: xr.Dataset):
        x = data.X.values
        y = data.Y.values

        delta = x[:, None] - x[None, :]
        K_base = np.exp(-0.5 * (delta / self.rho) ** 2)

        with pm.Model() as model:
            alpha = pm.HalfNormal("alpha", sigma=2.0)
            f_tilde = pm.Normal("f_tilde", mu=0.0, sigma=1.0, shape=self.n)

            cov = alpha ** 2 * K_base
            L = pt.linalg.cholesky(cov)
            f = pt.dot(L, f_tilde)

            # Manual numerically-stable Bernoulli logit likelihood.
            # pm.Bernoulli(logit_p=...) has unstable gradients for extreme logits.
            sign = 2.0 * y - 1.0
            pm.Potential("y_loglik", -pt.softplus(-sign * f).sum())

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "f_tilde": (["draw", "dim"], samples["f_tilde"]),
            },
            coords={
                "draw": np.arange(len(samples["alpha"])),
                "dim": np.arange(self.n),
            },
        )
