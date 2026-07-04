import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class SharedCholesky(BaseModel):
    """Nonlinear state-space model with two observation sequences sharing dynamics."""

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        return {
            "phi": np.array([0.5]),
            "sigma": np.array([1.0]),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 50,
        true_phi: float = 0.8,
        true_sigma: float = 0.5,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        def generate_seq(phi, sigma, n, rng):
            y = np.zeros(n)
            state = 0.0
            for t in range(n):
                state = np.tanh(phi * state)
                y[t] = state + sigma * rng.standard_normal()
            return y

        y1 = generate_seq(true_phi, true_sigma, n, rng)
        y2 = generate_seq(true_phi, true_sigma, n, rng)

        data = xr.Dataset(
            {
                "Y1": (["time"], y1),
                "Y2": (["time"], y2),
            },
            coords={"time": np.arange(n)},
            attrs={"n": n},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        y1 = test.Y1.values
        y2 = test.Y2.values
        n = len(y1)

        phi_samples = samples.phi.values
        sigma_samples = samples.sigma.values
        n_draws = len(phi_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            phi = phi_samples[i]
            sigma = np.abs(sigma_samples[i])
            ll = 0.0
            for y in [y1, y2]:
                state = 0.0
                for t in range(n):
                    state = np.tanh(phi * state)
                    r = y[t] - state
                    ll += -0.5 * (r / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
            pll[i] = ll

        return pll
