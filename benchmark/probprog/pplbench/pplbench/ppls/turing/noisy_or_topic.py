from typing import Dict

import numpy as np
import xarray as xr

from .base_turing_impl import BaseTuringImplementation


class NoisyOrTopic(BaseTuringImplementation):
    def __init__(self, **attrs) -> None:
        # train_data.attrs carries n, num_topics, num_words and edge_weight; the
        # Julia model reads them from the serialized data, so nothing is needed
        # here beyond accepting the keywords.
        self.attrs = attrs

    @property
    def julia_model_path(self) -> str:
        return "standard/noisy_or_topic"

    def extract_data_from_turing(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        # active recovered by argmax of the Gumbel-softmax relaxation on the Julia
        # side (approximate relaxation posterior, as upstream Stan's). Binary,
        # shape (draw, 1+num_topics), with the leak column always 1.
        active = np.asarray(samples["active"])
        return xr.Dataset(
            {"active": (["draw", "topic"], active.astype(int))},
            coords={
                "draw": np.arange(active.shape[0]),
                "topic": np.arange(active.shape[-1]),
            },
        )
