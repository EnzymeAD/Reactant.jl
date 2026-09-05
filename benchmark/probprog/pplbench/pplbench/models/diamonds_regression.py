# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class DiamondsRegression(BaseModel):
    """
    brms `diamonds` log-log price regression (posteriordb `diamonds-diamonds`).

    Model (posterior_database/models/stan/diamonds.stan, brms 2.10.0):

        Intercept ~ StudentT(3, 8, 10)
        b         ~ Normal(0, 1)^Kc              (Kc = K - 1 = 24)
        sigma     ~ HalfStudentT(3, 0, 10)
        Y         ~ Normal(Intercept + Xc @ b, sigma)

    where Xc holds the K - 1 non-intercept columns of the brms design matrix X,
    each centered at its own column mean (Stan's `transformed data` block).

    The benchmark folds the intercept back into the design matrix, fitting the
    identical likelihood as

        beta = [Intercept, b]        (K = 25)
        Y ~ Normal(X_aug @ beta, sigma),   X_aug = [1, Xc]

    which is the same model, the same parameters and the same priors, only
    written with one design matrix instead of a split intercept. Writing it this
    way is what a statistician writes down directly; the split in the Stan file
    is a brms code-generation artifact (it exists so `normal_id_glm_lpdf` can
    take the intercept as a separate argument).

    SICM: GaussianSufficientStats. The residual sum of squares
    ||Y - X_aug beta||^2 is answered from the thin QR of the sample-invariant
    X_aug, so the per-gradient cost drops from O(N K) plus several O(N)
    elementwise kernels to O(K^2).

    Data variables and dims:

        X[obs, feature]  - float, the augmented design [1, Xc], (N, K)
        Y[obs]           - float, log price

    Attributes: n (5000 observations), k (25 coefficients), kc (24).
    Lowercase because pplbench's own `simulate_data` reads `attrs["n"]` and
    `ppl_helper` splats the whole attrs dict into the PPL implementation's
    constructor and into `generate_init_params`.

    Posterior samples:

        beta[draw, feature] - float, (K,) = [Intercept, b]
        sigma[draw]         - float
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """Initial parameters in CONSTRAINED space."""
        rng = np.random.default_rng(seed)
        K = int(model_attrs.get("k", 25))
        # The intercept coordinate carries the response scale (Y is log price,
        # mean ~7.8); the slopes start near zero, as their Normal(0,1) prior
        # says. Drawing the intercept from its own prior location keeps both
        # arms' chains in the same basin from step one.
        beta0 = rng.uniform(-0.5, 0.5, size=(K,)).astype(np.float64)
        beta0[0] += 8.0
        return {
            "beta": beta0,
            "sigma": np.exp(rng.uniform(-1, 0, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 5000,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Load the recorded diamonds regression variables.

        Nothing is simulated: Y is the observed log price and X is the brms
        design matrix, both copied from posteriordb. `seed` is unused (there is
        no subsampling; see real_data/README.md, *Subsample policy*). `n` is
        accepted and ignored except as an assertion that the caller expected
        the full 5000 rows.
        """
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "real_data", "diamonds.json")
        with open(path) as f:
            raw = json.load(f)

        N = int(raw["N"])
        K = int(raw["K"])
        X_raw = np.asarray(raw["X"], dtype=np.float64)
        Y = np.asarray(raw["Y"], dtype=np.float64)
        assert X_raw.shape == (N, K)
        assert np.allclose(X_raw[:, 0], 1.0), "column 0 must be the ones column"
        if n and int(n) != N:
            raise ValueError(
                f"diamonds is used whole (N={N}); got n={n}")

        # Stan's `transformed data`: center every non-intercept column.
        Xc = X_raw[:, 1:] - X_raw[:, 1:].mean(axis=0, keepdims=True)
        X_aug = np.column_stack([np.ones(N), Xc])

        data = xr.Dataset(
            {
                "X": (["obs", "feature"], X_aug),
                "Y": (["obs"], Y),
            },
            coords={
                "obs": np.arange(N),
                "feature": np.arange(K),
            },
            attrs={"n": N, "k": K, "kc": K - 1},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """Per-draw predictive log likelihood of the held-out response."""
        X = test.X.values
        y = test.Y.values
        n = len(y)

        beta_samples = samples.beta.values
        sigma_samples = np.abs(samples.sigma.values)
        n_draws = len(sigma_samples)

        pll = np.zeros(n_draws)
        for i in range(n_draws):
            s = sigma_samples[i]
            resid = y - X @ beta_samples[i]
            pll[i] = -0.5 * (
                np.dot(resid, resid) / (s * s)
                + n * np.log(2 * np.pi * s * s)
            )
        return pll
