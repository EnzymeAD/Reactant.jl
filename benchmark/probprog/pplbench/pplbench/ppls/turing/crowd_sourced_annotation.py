from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class CrowdSourcedAnnotation(BaseTuringImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        num_categories: int,
        expected_correctness: float,
        num_labels_per_item: int,
        concentration: float,
        **kwargs,
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

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        prev = np.asarray(samples["prev"])  # (draw, num_categories)
        # (draw, k, true_category, obs_category)
        confusion_matrix = np.asarray(samples["confusion_matrix"])
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
