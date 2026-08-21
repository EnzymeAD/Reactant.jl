# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Crowd-Sourced Annotation (Dawid-Skene), ACL Q14-1025.
#
# Continuous latents: `prev` (K-simplex prevalence) and `confusion_matrix`
# (k*K independent K-simplices). The discrete per-item true label is
# marginalized analytically, matching the Stan and Impulse references so all
# three frameworks target the identical continuous density under NUTS:
#   per item i,  logsumexp_c ( sum_j log conf[labeler_ij, c, label_ij]
#                              + log prev[c] ).

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from jax.scipy.special import logsumexp

from .base_numpyro_impl import BaseNumPyroImplementation


class CrowdSourcedAnnotation(BaseNumPyroImplementation):
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

    def matched_init_params(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """Deterministic uniform-simplex init (matched-init control experiment).

        Constrained point identical to the Impulse arm's NUTS start
        (standard/crowd_sourced_annotation.jl `build_constraint`):
        `prev = ones(K)/K` and every confusion row `= ones(K)/K`. Under
        NumPyro's stick-breaking transform this is the unconstrained origin
        (z = 0), i.e. the symmetric point. Only used when the harness is
        invoked with --matched-init; the standard protocol never calls it.
        """
        K = self.num_categories
        k = self.k
        return {
            "prev": np.full((K,), 1.0 / K),
            "confusion_matrix": np.full((k, K, K), 1.0 / K),
        }

    def model(self, data: xr.Dataset):
        data = data.transpose("item", "item_label")
        labels = jnp.asarray(np.asarray(data.labels.values))      # (n, L) 0-based
        labelers = jnp.asarray(np.asarray(data.labelers.values))  # (n, L) 0-based
        K = self.num_categories
        k = self.k
        n, L = labels.shape

        # transformed data: Dirichlet concentrations (Stan `beta`, `alpha`).
        # beta[c] = 1/K (prevalence prior); alpha[c] is the confusion-row prior
        # for true category c: concentration*expected_correctness on the diagonal
        # entry, concentration*(1-expected_correctness)/(K-1) off-diagonal.
        beta = jnp.full(K, 1.0 / K)
        ec = self.expected_correctness
        off = (1.0 - ec) / (K - 1)
        alpha = self.concentration * (
            off * jnp.ones((K, K)) + (ec - off) * jnp.eye(K)
        )  # (K, K): alpha[c] is the concentration vector for true category c

        prev = numpyro.sample("prev", dist.Dirichlet(beta))  # (K,)
        with numpyro.plate("labeler", k, dim=-2):
            with numpyro.plate("true_category", K, dim=-1):
                # confusion_matrix[j, c, :] ~ Dirichlet(alpha[c]); the resulting
                # sample has shape (k, K, K) = (labeler, true, obs).
                confusion_matrix = numpyro.sample(
                    "confusion_matrix", dist.Dirichlet(alpha)
                )

        # Analytic marginalization of the discrete true label (matches Stan's
        # `target += log_sum_exp(log_prob)` over items).
        logconf = jnp.log(confusion_matrix)  # (k, K, K) = (labeler, true, obs)
        # gather logconf[labelers[i,j], c, labels[i,j]] for all (i, j, c)
        lc = logconf[
            labelers[:, :, None],
            jnp.arange(K)[None, None, :],
            labels[:, :, None],
        ]  # (n, L, K)
        per_item = lc.sum(axis=1)                       # (n, K) sum over labelers
        log_prob = per_item + jnp.log(prev)[None, :]    # (n, K)
        numpyro.factor("labels", logsumexp(log_prob, axis=1).sum())

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        prev = np.asarray(samples["prev"])  # (draw, K)
        # (draw, k, K, K) = (draw, labeler, true_category, obs_category)
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
