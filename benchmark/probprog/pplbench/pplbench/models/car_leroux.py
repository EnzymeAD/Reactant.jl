# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Leroux CAR disease-mapping model with the spatial-dependence parameter rho
ESTIMATED, as CARBayes fits it by default (Lee 2013, JSS 55(13),
`S.CARleroux(..., rho = NULL)`; Leroux, Lei and Breslow 2000):

    y_i       ~ Poisson(E_i exp(alpha + phi_i))
    phi       ~ MVN(0, (tau Q(rho))^{-1}),  Q(rho) = rho (D - W) + (1 - rho) I
    tau       = 1 / sigma_phi^2,             D = diag(rowsum W)
    alpha     ~ Normal(0, 10)
    sigma_phi ~ HalfNormal(1)
    rho       ~ Uniform(0, 1)

The fixed-rho companion is `car_spatial.py` (rho = 0.99, covariance form). This
class carries R = D - W as data; nothing depending on rho is precomputed.

Priors: alpha and sigma_phi as in car_spatial.py; rho ~ Uniform(0, 1) is
CARBayes's own prior. CARBayes's diffuse defaults for the intercept and the
variance (N(0, 1e5), InvGamma(1, 0.01)) are chosen for its Gibbs/Metropolis
updates and are not used here.

Data (see real_data/README.md), used whole, nothing simulated:
    scotland           Scotland lip cancer, 56 districts, 132 edges.
    nyc_traffic_n1921  posteriordb `traffic_accident_nyc` whole map, 1921
                       tracts, 5461 edges (the file the 2026-09-10 size sweep
                       built, real_data/nyc_traffic_submap_n1921.json).

SICM: CholeskyEigenLift on chol(tau rho R + tau (1 - rho) I), whose only use in
the density is the diagonal (0.5 log det of the precision).
"""
import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.special import gammaln

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)

REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "real_data"
)

REAL_DATASETS = {
    "scotland": "scotland_lip_cancer.json",
    "nyc_traffic_n1921": "nyc_traffic_submap_n1921.json",
}


class CARLeroux(BaseModel):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        # alpha, sigma_phi and phi are drawn exactly as
        # CARSpatial.generate_init_params draws them, so at a given seed they
        # equal the fixed-rho row's init; rho is drawn last.
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 50))
        out = {
            "alpha": rng.uniform(-1, 1, size=(1,)).astype(np.float64),
            "sigma_phi": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
            "phi": rng.standard_normal(n).astype(np.float64) * 0.1,
        }
        out["rho"] = rng.uniform(0.1, 0.9, size=(1,)).astype(np.float64)
        return out

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 56,
        dataset: str = "scotland",
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        # Real data only; `seed` is unused and `n` is cross-checked.
        assert dataset in REAL_DATASETS, dataset
        with open(os.path.join(REAL_DATA_DIR, REAL_DATASETS[dataset])) as f:
            raw = json.load(f)
        N = int(raw["N"])
        assert n == N, (n, N)
        W = np.asarray(raw["W"], dtype=np.float64)
        assert np.array_equal(W, W.T) and not np.any(np.diag(W))
        D = np.diag(np.sum(W, axis=1))
        R = D - W
        log_E = np.log(np.asarray(raw["E"], dtype=np.float64))
        y = np.asarray(raw["y"], dtype=np.float64)
        LOGGER.info(
            f"car_leroux[{dataset}]: n={N}, edges={int(W.sum() // 2)}, "
            f"min degree={int(W.sum(1).min())}, sum(y)={y.sum():.0f}"
        )
        data = xr.Dataset(
            {
                "R": (["i", "j"], R),
                "log_E": (["area"], log_E),
                "Y": (["area"], y),
            },
            coords={"i": np.arange(N), "j": np.arange(N), "area": np.arange(N)},
            attrs={"n": N},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        log_E = test.log_E.values
        y = test.Y.values.astype(int)
        alpha = np.asarray(samples.alpha.values).reshape(-1)
        phi = np.asarray(samples.phi.values)
        pll = np.zeros(len(alpha))
        for i in range(len(alpha)):
            log_rate = log_E + alpha[i] + phi[i]
            pll[i] = np.sum(y * log_rate - np.exp(log_rate) - gammaln(y + 1))
        return pll
