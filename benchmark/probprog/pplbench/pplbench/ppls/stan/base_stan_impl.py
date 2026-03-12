# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Dict, List

import xarray as xr

from ..base_ppl_impl import BasePPLImplementation


class BaseStanImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_code(self) -> str:
        """Get Stan code that represents the statistical model"""
        raise NotImplementedError

    @abstractmethod
    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        raise NotImplementedError

    @abstractmethod
    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """Convert the model data into format usable by Stan"""
        raise NotImplementedError

    @abstractmethod
    def extract_data_from_stan(self, params: Dict) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        raise NotImplementedError
