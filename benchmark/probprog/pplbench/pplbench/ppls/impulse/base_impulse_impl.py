from abc import abstractmethod
from typing import Dict

import xarray as xr

from ..base_ppl_impl import BasePPLImplementation


class BaseImpulseImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        ...

    @property
    @abstractmethod
    def julia_model_path(self) -> str:
        """Model path relative to benchmark dir (e.g. 'standard/logistic_regression')."""
        raise NotImplementedError

    @abstractmethod
    def extract_data_from_impulse(
        self, samples: Dict[str, list]
    ) -> xr.Dataset:
        raise NotImplementedError
