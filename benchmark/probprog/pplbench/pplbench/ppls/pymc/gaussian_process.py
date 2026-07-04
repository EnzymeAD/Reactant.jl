from typing import Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from .base_pymc_impl import BasePyMCImplementation


class GaussianProcess(BasePyMCImplementation):
    def __init__(self, n: int, noise_std: float) -> None:
        self.n = n
        self.noise_std = noise_std

    def build_model(self, data: xr.Dataset):
        X = data.X.values
        Y = data.Y.values

        with pm.Model() as model:
            kernel_var = pm.LogNormal("kernel_var", mu=0.0, sigma=10.0)
            kernel_length = pm.LogNormal("kernel_length", mu=0.0, sigma=10.0)
            kernel_noise = pm.LogNormal("kernel_noise", mu=0.0, sigma=10.0)

            # SE kernel
            delta = X[:, None] - X[None, :]
            K = kernel_var * pt.exp(-0.5 * (delta / kernel_length) ** 2)
            K = K + (kernel_noise + 1e-6) * pt.eye(self.n)

            pm.MvNormal("Y", mu=np.zeros(self.n), cov=K, observed=Y)

        return model

    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "kernel_var": (["draw"], samples["kernel_var"]),
                "kernel_length": (["draw"], samples["kernel_length"]),
                "kernel_noise": (["draw"], samples["kernel_noise"]),
            },
            coords={
                "draw": np.arange(len(samples["kernel_var"])),
            },
        )
