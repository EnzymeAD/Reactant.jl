import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class LoopSharedChol(BaseModel):
    """State-space model with shared Cholesky in the time loop."""

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        d = int(model_attrs.get("d", 5))
        T = int(model_attrs.get("T", 20))
        return {
            "sigma": np.array([1.0]),
            "x0": np.zeros(d),
            "x": np.zeros((T, d)),
        }

    @staticmethod
    def generate_data(
        seed: int, d: int = 5, T: int = 20, **kwargs,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Dense positive definite base matrix
        A = rng.standard_normal((d, d)) * 0.3
        Q_base = A @ A.T + np.eye(d)

        # Generate state-space data
        sigma_true = 1.0
        Q = sigma_true ** 2 * Q_base + np.eye(d)
        L = np.linalg.cholesky(Q)

        x = np.zeros((T, d))
        y = np.zeros((T, d))
        x_prev = np.zeros(d)
        for t in range(T):
            x[t] = x_prev + L @ rng.standard_normal(d)
            y[t] = x[t] + rng.standard_normal(d) * 0.1
            x_prev = x[t]

        data = xr.Dataset(
            {
                "Q_base": (["d_i", "d_j"], Q_base),
                "Y": (["time", "dim"], y),
            },
            coords={"time": np.arange(T), "dim": np.arange(d),
                     "d_i": np.arange(d), "d_j": np.arange(d)},
            attrs={"n": T * d, "d": d, "T": T},
        )
        # No train/test split for this model
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset,
    ) -> np.ndarray:
        # Simple placeholder — return zeros
        n_draws = samples.dims["draw"]
        return np.zeros(n_draws)
