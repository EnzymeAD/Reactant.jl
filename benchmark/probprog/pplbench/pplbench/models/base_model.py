# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import xarray as xr


class BaseModel(ABC):
    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        """
        Generate initial parameter values for MCMC inference.

        Called once per trial by PPLBench and passed to all backends so that
        every PPL starts from the same initial position.

        :param seed: A random number generator seed.
        :param model_attrs: Model-specific attributes (n, k, scales, etc.).
        :returns: A dict mapping parameter names to initial numpy arrays.
            Returns empty dict by default (backends use their own init strategy).
        """
        return {}

    @staticmethod
    @abstractmethod
    def generate_data(seed: int, n: int, **model_args) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Generate data from the model by forward sampling from the prior distribution.
        :param seed: A random number generator seed.
        :param n: The number of items in the dataset.
        :returns: A tuple of training and test datasets.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate_posterior_predictive(
        post_samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Evaluate and return the predictive log likelihood (PLL).
        :param post_samples: The posterior samples from a single MCMC chain.
            The `post_samples` object must have a `sample` dimension.
        :param test: The test data previously created by `generate_data`.
        :returns: The PLL of each sample on the test data.
        """
        raise NotImplementedError

    @staticmethod
    def additional_metrics(
        output_dir: str, post_samples: xr.Dataset, train: xr.Dataset, test: xr.Dataset
    ) -> None:
        """
        Compute model-specific metrics other than the standard PLL
        and write them out. This method is optional.
        :param output_dir: directory to store the results
        :post_samples: all posterior samples; with coords (ppl, chain, draw, ...)
        :train: training dataset
        :test: test dataset
        """
        pass
