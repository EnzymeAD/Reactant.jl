# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Dict

import jax.numpy as jnp
import xarray as xr

from ..base_ppl_impl import MATCHED_INIT_REAL, BasePPLImplementation

# Re-exported so numpyro model impls can import it alongside the base class.
# The constant and its full rationale live in ppls/base_ppl_impl.py; every arm
# (NumPyro, Stan, Impulse harness.jl, Turing turing_harness.jl) must pin the
# same literal or the matched-init protocol stops comparing a shared point.
__all__ = ["BaseNumPyroImplementation", "MATCHED_INIT_REAL"]


class BaseNumPyroImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        """
        :param attrs: model arguments
        """
        ...

    @abstractmethod
    def model(self, data: xr.Dataset):
        """
        A python callable object with NumPyro primitives that
        represents the statistical model of interest.

        :param data: the inputs to the model including observed data.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        """
        Takes the output of NumPyro inference and converts into a format expected
        by PPLBench.
        :param samples: A dict of samples keyed by names of latent variables in
            the model.
        :returns: a dataset over inferred parameters
        """
        raise NotImplementedError
