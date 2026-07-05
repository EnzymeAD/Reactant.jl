# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.special import gammaln, logsumexp

from .base_model import BaseModel


class CoalMining(BaseModel):
    """
    Coal-Mining Disasters Change-Point Model

    A Poisson time series whose rate jumps at an unknown switch year ``s``.
    Canonical PyMC/Stan change-point model. Synthetic data with a known
    ``s_true`` gives a ground-truth change point for the recovery figure.

    Model (marginalized over the discrete switch ``s``):

        l1 ~ Gamma(3, 1)                       # early (pre-switch) rate
        l2 ~ Gamma(3, 1)                       # late  (post-switch) rate
        s  ~ DiscreteUniform{0, ..., T}        # switch position (marginalized)
        for t in 0 .. T-1:
            y_t ~ Poisson(t < s ? l1 : l2)

    The discrete ``s`` is integrated out exactly with the prefix/suffix
    cumulative-sum trick (Stan-canonical):

        a_t = poisson_logpmf(y_t | l1),  b_t = poisson_logpmf(y_t | l2)
        A_s = sum_{t<s} a_t   (cumsum, A_0 = 0)
        B_s = sum_{t>=s} b_t  (reverse cumsum, B_T = 0)
        lp_s = -log(T+1) + A_s + B_s
        log p(y | l1, l2) = logsumexp_s lp_s

    Continuous latents l1, l2 are inferred in log space (positive support).

    The dataset has a single variable ``y`` over the ``year`` dimension and
    the attributes T, s_true, l1_true, l2_true. Train and test are both the
    full series: a change-point time series does not split i.i.d., so the
    per-draw predictive metric is the in-sample marginal log-likelihood
    logsumexp_s(lp_s) on the full series.
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """
        Init l1, l2 in constrained (positive) space as exp(Uniform(-2, 2)),
        so the unconstrained log-rates start at Uniform(-2, 2), matching
        NumPyro's default ``init_to_uniform(radius=2)``.
        """
        rng = np.random.default_rng(seed)
        return {
            "l1": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "l2": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
        }

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 112,
        s_true: int = 40,
        l1_true: float = 3.0,
        l2_true: float = 0.9,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        :param n: length of the series (default 112, the coal-mining span).
        :param s_true: true switch position (default 40).
        :param l1_true, l2_true: true pre/post-switch Poisson rates.
        :param train_frac: unused; the full series is both train and test.
        """
        rng = np.random.default_rng(seed)
        rates = np.where(np.arange(n) < s_true, l1_true, l2_true)
        y = rng.poisson(rates).astype(np.int64)

        data = xr.Dataset(
            {"y": (["year"], y)},
            coords={"year": np.arange(n)},
            attrs={
                "n": n,
                "s_true": s_true,
                "l1_true": l1_true,
                "l2_true": l2_true,
            },
        )
        # No held-out split: the metric is the in-sample marginal.
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        In-sample marginal log-likelihood per posterior draw:
        logsumexp_s(lp_s) on the full series, for each (l1, l2) draw.
        :returns: a numpy array of length = number of draws.
        """
        l1 = np.asarray(samples.l1.values).reshape(-1)  # (draws,)
        l2 = np.asarray(samples.l2.values).reshape(-1)  # (draws,)
        y = np.asarray(test.y.values).reshape(-1)       # (T,)
        T = y.shape[0]
        lgam = gammaln(y + 1.0)                          # (T,)

        # a, b : (draws, T)
        a = y[None, :] * np.log(l1)[:, None] - l1[:, None] - lgam[None, :]
        b = y[None, :] * np.log(l2)[:, None] - l2[:, None] - lgam[None, :]

        draws = l1.shape[0]
        zeros = np.zeros((draws, 1))
        # A_s = sum_{t<s} a_t, s = 0..T   (leading zero + cumsum)
        A = np.concatenate([zeros, np.cumsum(a, axis=1)], axis=1)  # (draws, T+1)
        # B_s = sum_{t>=s} b_t, s = 0..T  (reverse cumsum + trailing zero)
        B = np.concatenate(
            [np.cumsum(b[:, ::-1], axis=1)[:, ::-1], zeros], axis=1
        )  # (draws, T+1)
        lp = -np.log(T + 1.0) + A + B                    # (draws, T+1)
        return np.asarray(logsumexp(lp, axis=1))         # (draws,)
