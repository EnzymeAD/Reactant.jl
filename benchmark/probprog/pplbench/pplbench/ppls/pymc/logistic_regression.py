from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class LogisticRegression(BasePyMCImplementation):
    def __init__(
        self, n: int, k: int, alpha_scale: float, beta_scale: float, beta_loc: float
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc

    def build_model(self, data: xr.Dataset):
        data = data.transpose("item", "feature")
        X, Y = data.X.values, data.Y.values

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=self.alpha_scale)
            beta = pm.Normal("beta", mu=self.beta_loc, sigma=self.beta_scale,
                             shape=self.k)
            mu = alpha + pt.dot(X, beta)
            # Manual numerically-stable Bernoulli logit likelihood.
            # pm.Bernoulli(logit_p=...) has unstable gradients for extreme logits.
            sign = 2.0 * Y - 1.0
            pm.Potential("Y_loglik", -pt.softplus(-sign * mu).sum())

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
