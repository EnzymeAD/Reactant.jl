# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class ScaledMultiSensor(BaseModel):
    """
    Scaled Multi-Sensor Fusion (DotGeneralScale benchmark).

    Model:
        sigma ~ HalfNormal(1)
        f_tilde ~ Normal(0, 1)^n
        z_k = (sigma * M_k) @ s_k  for k = 1..5
        pred = sum(z_k)
        y ~ Normal(pred + f_tilde, 1)

    SICM: DotGeneralScaleFactorization hoists M_k @ s_k.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 200))
        return {
            "sigma": np.array([1.0]),
            "f_tilde": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 200,
        d_state: int = 200,
        true_sigma: float = 2.0,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Generate 5 random sensor matrices (n × d_state) and signals (d_state × 1)
        Ms = [rng.standard_normal((n, d_state)) * 0.01 for _ in range(5)]
        ss = [rng.standard_normal((d_state, 1)) for _ in range(5)]

        # True prediction
        pred = sum(true_sigma * M @ s for M, s in zip(Ms, ss)).ravel()
        # Observations
        y = pred + rng.standard_normal(n)

        data_vars = {"Y": (["point"], y)}
        for k in range(5):
            data_vars[f"M{k+1}"] = (["obs", f"state_{k}"], Ms[k])
            data_vars[f"s{k+1}"] = ([f"state_{k}", f"col_{k}"], ss[k])

        data = xr.Dataset(
            data_vars,
            coords={"point": np.arange(n)},
            attrs={"n": n, "d_state": d_state},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        n = test.attrs["n"]
        d_state = test.attrs["d_state"]
        y = test.Y.values

        sigma_samples = samples.sigma.values
        f_tilde_samples = samples.f_tilde.values
        n_draws = len(sigma_samples)

        Ms = [test[f"M{k+1}"].values for k in range(5)]
        ss = [test[f"s{k+1}"].values for k in range(5)]

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            sigma_i = np.atleast_1d(sigma_samples[i]).item()
            f_tilde_i = f_tilde_samples[i]
            pred = sum(sigma_i * M @ s for M, s in zip(Ms, ss)).ravel()
            mu = pred + f_tilde_i
            pll[i] = -0.5 * np.sum((y - mu) ** 2)

        return pll
