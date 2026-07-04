from abc import abstractmethod
from typing import Dict

import xarray as xr

from ..base_ppl_impl import BasePPLImplementation


class BaseTuringImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        ...

    @property
    @abstractmethod
    def julia_model_path(self) -> str:
        """Model path relative to turing/ dir (e.g. 'standard/logistic_regression')."""
        raise NotImplementedError

    @abstractmethod
    def extract_data_from_turing(
        self, samples: Dict[str, list]
    ) -> xr.Dataset:
        raise NotImplementedError
