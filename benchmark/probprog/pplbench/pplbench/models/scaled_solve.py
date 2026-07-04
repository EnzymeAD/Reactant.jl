# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class ScaledSolve(BaseModel):
    """
    Scaled triangular solve benchmark (TriSolveScale, no Cholesky).
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
        true_sigma: float = 2.0,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        # Generate a well-conditioned lower-triangular matrix
        A = rng.standard_normal((n, n)) * 0.1
        A += np.eye(n) * 2.0  # diagonal dominance
        L_base = np.linalg.cholesky(A @ A.T + np.eye(n))

        # 5 data vectors
        xs = [rng.standard_normal((n, 1)) for _ in range(5)]

        # True prediction
        L_true = true_sigma * L_base
        pred = sum(np.linalg.solve(L_true, x) for x in xs).ravel()
        y = pred + rng.standard_normal(n)

        data_vars = {
            "L_base": (["i", "j"], L_base),
            "Y": (["point"], y),
        }
        for k in range(5):
            data_vars[f"x{k+1}"] = ([f"row_{k}", f"col_{k}"], xs[k])

        data = xr.Dataset(
            data_vars,
            coords={"point": np.arange(n)},
            attrs={"n": n},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        return np.zeros(len(samples.sigma.values))
