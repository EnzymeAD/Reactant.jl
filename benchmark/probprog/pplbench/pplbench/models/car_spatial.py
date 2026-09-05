# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

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

# Real areal-count data sets this model can be fitted to. Both files carry
# N, y (observed counts), E (exposure / expected counts) and W (dense 0/1
# adjacency); see real_data/README.md for provenance.
REAL_DATASETS = {
    "scotland": "scotland_lip_cancer.json",
    "nyc_traffic": "nyc_traffic_submap.json",
}


def _random_adjacency(rng, n, avg_neighbors=4):
    """Generate a random sparse symmetric adjacency matrix."""
    # Create a random geometric graph
    locs = rng.uniform(0, 1, size=(n, 2))
    dist = np.sqrt(np.sum((locs[:, None, :] - locs[None, :, :]) ** 2, axis=-1))
    # Choose threshold to get ~avg_neighbors per node
    threshold = np.sort(dist.ravel())[n * avg_neighbors]
    W = (dist < threshold).astype(float)
    np.fill_diagonal(W, 0.0)
    # Ensure symmetric
    W = np.maximum(W, W.T)
    return W


class CARSpatial(BaseModel):
    """
    Leroux CAR: y_i ~ Poisson(E_i * exp(alpha + phi_i))
    phi ~ MVN(0, sigma_phi^2 * Q_inv)
    Q = rho*(D-W) + (1-rho)*I, precomputed.

    SICM: ScaleFamily cascade on sigma_phi^2 * Q_inv.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = int(model_attrs.get("n", 50))
        return {
            "alpha": rng.uniform(-1, 1, size=(1,)).astype(np.float64),
            "sigma_phi": np.exp(rng.uniform(-1, 0.5, size=(1,))).astype(np.float64),
            "phi": rng.standard_normal(n).astype(np.float64) * 0.1,
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 50,
        rho: float = 0.99,
        true_alpha: float = 0.5,
        true_sigma_phi: float = 0.8,
        avg_neighbors: int = 4,
        train_frac: float = 1.0,
        dataset: str = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)

        if dataset in REAL_DATASETS:
            # Real data. Both branches carry the observed count vector as Y and
            # the recorded exposure/expected counts as the offset; `seed` is
            # unused (nothing is simulated). See real_data/README.md.
            #   scotland    Scotland lip cancer, 56 districts (Clayton & Kaldor
            #               1987; Breslow & Clayton 1993; BUGS examples), from
            #               the Stan ICAR case study data file.
            #   nyc_traffic 250-tract contiguous sub-map of the posteriordb
            #               `traffic_accident_nyc` data (1921 tracts).
            with open(os.path.join(REAL_DATA_DIR, REAL_DATASETS[dataset])) as f:
                raw = json.load(f)
            n = int(raw["N"])
            W = np.asarray(raw["W"], dtype=np.float64)
            D = np.diag(np.sum(W, axis=1))

            # Same Leroux construction as the synthetic branch below.
            Q = rho * (D - W) + (1 - rho) * np.eye(n)
            Q_inv = np.linalg.inv(Q)
            Q_inv = 0.5 * (Q_inv + Q_inv.T)
            Q_inv += 1e-6 * np.eye(n)

            log_E = np.log(np.asarray(raw["E"], dtype=np.float64))
            y = np.asarray(raw["y"], dtype=np.float64)

            LOGGER.info(
                f"car_spatial[{dataset}]: n={n}, edges={int(W.sum() // 2)}, "
                f"cond(Q)={np.linalg.cond(Q):.4g}, sum(y)={y.sum():.0f}"
            )

            data = xr.Dataset(
                {
                    "Q_inv": (["i", "j"], Q_inv),
                    "log_E": (["area"], log_E),
                    "Y": (["area"], y),
                },
                coords={
                    "i": np.arange(n),
                    "j": np.arange(n),
                    "area": np.arange(n),
                },
                attrs={"n": n, "rho": rho},
            )
            return data, data

        # Adjacency matrix
        W = _random_adjacency(rng, n, avg_neighbors)
        D = np.diag(np.sum(W, axis=1))

        # Leroux precision: Q = rho*(D-W) + (1-rho)*I
        Q = rho * (D - W) + (1 - rho) * np.eye(n)
        Q_inv = np.linalg.inv(Q)
        # Ensure symmetry
        Q_inv = 0.5 * (Q_inv + Q_inv.T)
        Q_inv += 1e-6 * np.eye(n)

        # Expected counts (from population/demographics)
        E = rng.uniform(50, 200, n)
        log_E = np.log(E)

        # Generate spatial effects
        Sigma_phi = true_sigma_phi ** 2 * Q_inv
        L = np.linalg.cholesky(Sigma_phi)
        phi = L @ rng.standard_normal(n)

        # Generate Poisson counts
        log_rate = log_E + true_alpha + phi
        y = rng.poisson(np.exp(log_rate)).astype(float)

        data = xr.Dataset(
            {
                "Q_inv": (["i", "j"], Q_inv),
                "log_E": (["area"], log_E),
                "Y": (["area"], y),
            },
            coords={
                "i": np.arange(n),
                "j": np.arange(n),
                "area": np.arange(n),
            },
            attrs={"n": n, "rho": rho},
        )

        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        log_E = test.log_E.values
        y = test.Y.values.astype(int)
        n = len(y)

        alpha_samples = samples.alpha.values
        phi_samples = samples.phi.values  # n_draws x n
        n_draws = len(alpha_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            alpha_i = alpha_samples[i]
            phi_i = phi_samples[i]
            log_rate = log_E + alpha_i + phi_i
            rate = np.exp(log_rate)
            # Poisson log-likelihood: y*log(rate) - rate - log(y!)
            pll[i] = np.sum(y * np.log(rate + 1e-30) - rate - gammaln(y + 1))

        return pll
