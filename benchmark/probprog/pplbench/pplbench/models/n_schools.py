# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.stats import norm

from .base_model import BaseModel


LOGGER = logging.getLogger(__name__)


class NSchools(BaseModel):
    """
    N Schools

    This is a generalization of a classical 8 schools model to n schools.
    The model posits that the effect of a school on a student's performance
    can be explained by the a baseline effect of all schools plus an additive
    effect of the state, the school district and the school type.

    Hyper Parameters:

        n - total number of schools
        num_states - number of states
        num_districts_per_state - number of school districts in each state
        num_types - number of school types
        scale_state - state effect scale
        scale_district - district effect scale
        scale_type - school type effect scale

    Model:


        beta_baseline = StudentT(dof_baseline, 0.0, scale_baseline)

        sigma_state ~ HalfCauchy(0, scale_state)

        sigma_district ~ HalfCauchy(0, scale_district)

        sigma_type ~ HalfCauchy(0, scale_type)

        for s in 0 .. num_states - 1
            beta_state[s] ~ Normal(0, sigma_state)

            for d in 0 .. num_districts_per_state - 1
                beta_district[s, d] ~ Normal(0, sigma_district)

        for t in 0 .. num_types - 1
            beta_type[t] ~ Normal(0, sigma_type)

        for i in 0 ... n - 1
            Assume we are given state[i], district[i], type[i]

            Y_hat[i] = beta_baseline + beta_state[state[i]]
                        + beta_district[state[i], district[i]]
                        + beta_type[type[i]]

            sigma[i] ~ Uniform(0.5, 1.5)

            Y[i] ~ Normal(Y_hat[i], sigma[i])

    The dataset consists of the following

        Y[school]         - float
        sigma[school]     - float

    and it includes the attributes

        n  - number of schools
        num_states
        num_districts_per_state
        num_types
        dof_baseline
        scale_baseline
        scale_state
        scale_district
        scale_type
        state_idx[school]     - 0 .. num_states - 1
        district_idx[school]  - 0 .. num_districts_per_state - 1
        type_idx[school]      - 0 .. num_types - 1

    The posterior samples include the following,

        sigma_state[draw]                    - float
        sigma_district[draw]                 - float
        sigma_type[draw]                     - float
        beta_baseline[draw]                  - float
        beta_state[draw, state]              - float
        beta_district[draw, state, district] - float
        beta_type[draw, type]                - float
    """

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        num_states = int(model_attrs.get("num_states", 8))
        num_districts_per_state = int(model_attrs.get("num_districts_per_state", 5))
        num_types = int(model_attrs.get("num_types", 5))
        return {
            "sigma_state": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "sigma_district": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "sigma_type": np.exp(rng.uniform(-2, 2, size=(1,))).astype(np.float64),
            "beta_baseline": rng.uniform(-2, 2, size=(1,)).astype(np.float64),
            "beta_state": rng.uniform(-2, 2, size=(num_states,)).astype(np.float64),
            "beta_district_flat": rng.uniform(-2, 2, size=(num_states * num_districts_per_state,)).astype(np.float64),
            "beta_type": rng.uniform(-2, 2, size=(num_types,)).astype(np.float64),
        }

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2000,
        num_states: int = 8,
        num_districts_per_state: int = 5,
        num_types: int = 5,
        dof_baseline: float = 3.0,
        scale_baseline: float = 10.0,
        scale_state: float = 1.0,
        scale_district: float = 1.0,
        scale_type: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        See the class documentation for an explanation of the parameters.
        :param seed: random number generator seed
        """
        if n % 2 != 0:
            LOGGER.warn(f"n should be a multiple of 2. Actual values = {n}")
        # In this model we will generate exactly equal amounts of training
        # and test data with the same number of training and test schools
        # in each state, district, and type combination
        n = n // 2
        rng = np.random.default_rng(seed)
        beta_baseline = rng.standard_t(dof_baseline) * scale_baseline
        sigma_state = np.abs(rng.standard_cauchy()) * scale_state
        sigma_district = np.abs(rng.standard_cauchy()) * scale_district
        sigma_type = np.abs(rng.standard_cauchy()) * scale_type
        beta_state = rng.normal(loc=0, scale=sigma_state, size=num_states)
        beta_district = rng.normal(
            loc=0, scale=sigma_district, size=(num_states, num_districts_per_state)
        )
        beta_type = rng.normal(loc=0, scale=sigma_type, size=num_types)

        # we will randomly assign the schools to states, district, and types
        state_idx = rng.integers(low=0, high=num_states, size=n)
        district_idx = rng.integers(low=0, high=num_districts_per_state, size=n)
        type_idx = rng.integers(low=0, high=num_types, size=n)

        y_hat = (
            beta_baseline
            + beta_state[state_idx]
            + beta_district[state_idx, district_idx]
            + beta_type[type_idx]
        )
        train_sigma = rng.uniform(0.5, 1.5, size=n)
        train_y = rng.normal(loc=y_hat, scale=train_sigma)
        test_sigma = rng.uniform(0.5, 1.5, size=n)
        test_y = rng.normal(loc=y_hat, scale=test_sigma)

        return tuple(  # type: ignore
            xr.Dataset(
                {"Y": (["school"], y), "sigma": (["school"], sigma)},
                coords={"school": np.arange(n)},
                attrs={
                    "n": n,
                    "num_states": num_states,
                    "num_districts_per_state": num_districts_per_state,
                    "num_types": num_types,
                    "dof_baseline": dof_baseline,
                    "scale_baseline": scale_baseline,
                    "scale_state": scale_state,
                    "scale_district": scale_district,
                    "scale_type": scale_type,
                    "state_idx": state_idx,
                    "district_idx": district_idx,
                    "type_idx": type_idx,
                },
            )
            for y, sigma in [(train_y, train_sigma), (test_y, test_sigma)]
        )

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Computes the predictive likelihood of all the test items w.r.t. each sample.
        See the class documentation for the `samples` and `test` parameters.
        :returns: a numpy array of the same size as the sample dimension.
        """
        # transpose the datasets to be in a convenient format
        samples = samples.transpose("draw", "state", "district", "type")
        y_hat = (
            samples.beta_baseline.values[:, np.newaxis]
            + samples.beta_state.values[:, test.attrs["state_idx"]]
            + samples.beta_district.values[
                :, test.attrs["state_idx"], test.attrs["district_idx"]
            ]
            + samples.beta_type.values[:, test.attrs["type_idx"]]
        )  # size = (iterations, n_test)
        loglike = norm.logpdf(
            test.Y.values[np.newaxis, :],
            loc=y_hat,
            scale=test.sigma.values[np.newaxis, :],
        )  # size = (iterations, n_test)
        return loglike.sum(axis=1)  # size = (iterations,)
