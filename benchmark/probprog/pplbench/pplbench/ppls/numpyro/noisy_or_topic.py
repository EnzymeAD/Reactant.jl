# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Noisy-Or Topic model (Latent Keyphrase Inference / LAKI), NumPyro arm.
#
# Port of the upstream Stan noisy_or_topic Gumbel-softmax continuous RELAXATION
# (temperature tau=0.1), matching the Stan and Impulse references so the three
# continuous PPLs target the identical APPROXIMATE relaxation posterior under
# NUTS (it is NOT an exact marginalization of the discrete topic states). Two
# Gumbel(0,1) latent vectors (`true` / `false`) of length 1+num_topics (the leak
# entry is sampled but unused, pinned active). The sequential noisy-OR recursion
# follows the Stan `transformed parameters` block exactly; num_topics is static
# so a plain Python loop over the topological order traces cleanly under JIT.
# Per topic j:
#   aw_j           = edge_weight[j, :j] . active_true[:j]
#   active_true[j] = softmax([log1m_exp(-aw_j)+g_true[j], -aw_j+g_false[j]] / tau)[0]
#                  = sigmoid(((log1m_exp(-aw_j)+aw_j) + (g_true[j]-g_false[j])) / tau).
# Word logits reuse the solved vector:
#   S_weight = edge_weight[words, :] . active_true
#   S_logit  = log1m_exp(-S_weight) + S_weight;  S ~ Bernoulli(logits = S_logit).
# The boolean `active` is recovered by argmax (equivalently lt > lf) in the
# extract step, exactly as the Stan extract does.

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from .base_numpyro_impl import BaseNumPyroImplementation


def _log1mexpm(x):
    # log(1 - exp(-x)) for x >= 0, matching the Stan log1m_exp(-x) and the
    # Impulse arm's _log1mexpm(x) = log(-expm1(-x)).
    return jnp.log(-jnp.expm1(-x))


class NoisyOrTopic(BaseNumPyroImplementation):
    def __init__(self, n, num_topics, num_words, edge_weight, **kwargs) -> None:
        self.n = n
        self.num_topics = num_topics
        self.num_words = num_words
        self.edge_weight = np.asarray(edge_weight)
        self.tau = 0.1

    def matched_init_params(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """Deterministic canonical init (matched-init control experiment).

        Constrained point identical to the Impulse arm's NUTS start
        (standard/noisy_or_topic.jl `build_constraint`): both Gumbel latent
        vectors at 0. Gumbel support is the whole real line (identity transform),
        so this is the unconstrained origin (z = 0). Routed through init_to_value
        by numpyro_harness.py; only used with --matched-init.
        """
        P = 1 + self.num_topics
        return {
            "active_gumbel_true": np.zeros(P),
            "active_gumbel_false": np.zeros(P),
        }

    def model(self, data: xr.Dataset):
        data = data.transpose("sentence", "word")
        # Training sentence(s): n==2 with train_frac 0.5 -> one train sentence.
        S = jnp.asarray(np.asarray(data.S.values)[0])   # (num_words,)
        edge_weight = jnp.asarray(self.edge_weight)
        P = 1 + self.num_topics
        tau = self.tau

        # Gumbel(0,1) latents, `true`/`false`, length P (leak entry unused).
        g_true = numpyro.sample(
            "active_gumbel_true", dist.Gumbel(0.0, 1.0).expand([P]).to_event(1)
        )
        g_false = numpyro.sample(
            "active_gumbel_false", dist.Gumbel(0.0, 1.0).expand([P]).to_event(1)
        )

        # Sequential noisy-OR recursion over the topological order (leak pinned).
        active = [jnp.asarray(1.0)]                     # active_true[0] = leak = 1
        for j in range(1, P):
            aw = jnp.dot(edge_weight[j, :j], jnp.stack(active))     # aw_j
            lt = (_log1mexpm(aw) + g_true[j]) / tau
            lf = (-aw + g_false[j]) / tau
            active.append(jax.nn.sigmoid(lt - lf))       # softmax "true" prob
        active_vec = jnp.stack(active)                   # (P,)

        # Word logits reuse the solved relaxed vector.
        word_edges = edge_weight[P:P + self.num_words, :]  # (num_words, P)
        S_weight = word_edges @ active_vec
        S_logit = _log1mexpm(S_weight) + S_weight
        numpyro.sample("S", dist.Bernoulli(logits=S_logit).to_event(1), obs=S)

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        g_true = np.asarray(samples["active_gumbel_true"])   # (draw, P)
        g_false = np.asarray(samples["active_gumbel_false"])
        edge_weight = np.asarray(self.edge_weight)
        tau = self.tau
        ndraws, P = g_true.shape

        # Recover boolean `active` by the sequential recursion (argmax over the
        # 2-way relaxation), matching the Stan/Impulse extracts. aw uses the soft
        # relaxed active_true; the hard decision lt > lf is tau-invariant.
        active = np.zeros((ndraws, P), dtype=int)
        active[:, 0] = 1                                  # leak
        a1 = np.ones((ndraws, P))                         # relaxed active_true
        for j in range(1, P):
            aw = a1[:, :j] @ edge_weight[j, :j]           # (ndraws,)
            lt = np.log(-np.expm1(-aw)) + g_true[:, j]
            lf = -aw + g_false[:, j]
            a1[:, j] = 1.0 / (1.0 + np.exp(-(lt - lf) / tau))
            active[:, j] = (lt > lf).astype(int)

        return xr.Dataset(
            {"active": (["draw", "topic"], active)},
            coords={
                "draw": np.arange(ndraws),
                "topic": np.arange(P),
            },
        )
