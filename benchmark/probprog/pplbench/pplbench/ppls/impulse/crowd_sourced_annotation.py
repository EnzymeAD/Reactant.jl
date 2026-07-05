from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class CrowdSourcedAnnotation(BaseImpulseImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        num_categories: int,
        expected_correctness: float,
        num_labels_per_item: int,
        concentration: float,
    ) -> None:
        self.n = n
        self.k = k
        self.num_categories = num_categories
        self.expected_correctness = expected_correctness
        self.num_labels_per_item = num_labels_per_item
        self.concentration = concentration

    @property
    def julia_model_path(self) -> str:
        return "standard/crowd_sourced_annotation"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        prev = samples["prev"]  # (draw, num_categories)
        confusion_matrix = samples["confusion_matrix"]  # (draw, k, true, obs)
        return xr.Dataset(
            {
                "prev": (["draw", "true_category"], prev),
                "confusion_matrix": (
                    ["draw", "labeler", "true_category", "obs_category"],
                    confusion_matrix,
                ),
            },
            coords={
                "draw": np.arange(prev.shape[0]),
                "true_category": np.arange(prev.shape[-1]),
                "obs_category": np.arange(prev.shape[-1]),
                "labeler": np.arange(confusion_matrix.shape[1]),
            },
        )
