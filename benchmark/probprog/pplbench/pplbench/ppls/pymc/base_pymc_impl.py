from abc import abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

from ..base_ppl_impl import BasePPLImplementation


class BasePyMCImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        """
        :param model_attrs: model arguments
        """
        ...

    @abstractmethod
    def build_model(self, data: xr.Dataset):
        """
        Build and return a PyMC model object.

        :param data: the inputs to the model including observed data.
        :returns: a pm.Model instance
        """
        raise NotImplementedError

    @abstractmethod
    def extract_data_from_pymc(
        self, samples: Dict[str, np.ndarray]
    ) -> xr.Dataset:
        """
        Takes the output of PyMC inference and converts into a format expected
        by PPLBench.
        :param samples: A dict of samples keyed by names of latent variables.
        :returns: a dataset over inferred parameters
        """
        raise NotImplementedError
