# Impulse implementation of PGLS with Pagel's lambda estimated
# (models/pgls_lambda.py). Runs the Julia model sicm/pgls_lambda.jl through
# BaseImpulseImplementation and returns the draws of beta, sigma and lambda as an
# xarray Dataset.
from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class PGLSLambda(BaseImpulseImplementation):
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k

    @property
    def julia_model_path(self) -> str:
        return "sicm/pgls_lambda"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta": (["draw", "feature"], np.asarray(samples["beta"])),
                "sigma": (["draw"], np.asarray(samples["sigma"]).reshape(-1)),
                "lambda": (["draw"], np.asarray(samples["lambda"]).reshape(-1)),
            },
            coords={
                "draw": np.arange(len(samples["sigma"])),
                "feature": np.arange(self.k),
            },
        )
