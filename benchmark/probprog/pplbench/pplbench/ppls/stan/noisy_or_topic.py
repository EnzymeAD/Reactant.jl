# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Upstream facebookresearch/pplbench Stan implementation, with two fork
# adaptations (mirroring stan/crowd_sourced_annotation):
#   (a) conditional chain-dim squeeze in extract (the fork calls extract twice:
#       in the harness and on the reloaded JSON, so an unconditional squeeze(1)
#       breaks the second call);
#   (b) the Stan program modernized to array[...] declarations (cmdstan 2.38
#       removed the pre-2.33 `int S[num_words]` syntax). Model math is identical
#       to upstream: the Gumbel-softmax continuous relaxation (tau=0.1), an
#       APPROXIMATE posterior, not an exact marginalization of the discrete
#       topic states.

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class NoisyOrTopic(BaseStanImplementation):
    def __init__(self, **attrs) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def matched_init_params(self, data: xr.Dataset) -> Dict:
        """Deterministic canonical init (matched-init control experiment).

        Constrained values identical to the Impulse arm's NUTS start: the
        `active_gumbel` matrix ((1+num_topics) x 2) is all zeros. Gumbel support
        is the whole real line (identity transform), so constrained 0 ==
        unconstrained 0. Stan inits live on the constrained scale, so this is
        passed straight through as `inits`. Only used with --matched-init.
        """
        P = 1 + int(self.attrs["num_topics"])
        return {
            "active_gumbel": np.zeros((P, 2)),
        }

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("sentence", "word")
        assert self.attrs["n"] == 1
        self.attrs["S"] = data.S.values[0]
        self.attrs["tau"] = 0.1
        # note: attrs should already contain n, edge_weight, num_topics and num_words
        return self.attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        active = np.asarray(samples["active"])
        # The fork calls extract twice: first on the raw Stan output
        # [iters, (chain,) topic, 2], then again on the reloaded JSON which is
        # already the recovered binary [iters, topic]. Only argmax-recover when
        # the size-2 relaxation axis is still present.
        if active.ndim >= 3:
            if active.ndim > 3:
                active = active.squeeze(1)  # drop chain dim
            active = (np.argmax(active, axis=2) == 0)  # [iters, num_topics]
        active = active.astype(int)
        return xr.Dataset(
            {"active": (["draw", "topic"], active.astype(int))},
            coords={
                "draw": np.arange(active.shape[0]),
                "topic": np.arange(active.shape[-1]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["active"]

    def get_code(self) -> str:
        """
        This implementation relies on the Gumbel softmax trick
        https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
        We represent each active node as a simplex which tends to take on
        extreme values. So, true should [1, 0] and false sould be [0, 1]
        ideally in a 1-hot representation, but in a gumbel softmax
        representation true would look more like [.99999, 1e-10] for example.

        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
  int<lower=1, upper=1> n;
  int<lower = 1> num_topics;
  int<lower = 1> num_words;
  matrix<lower = 0> [1 + num_topics + num_words, 1 + num_topics] edge_weight;
  real tau;
  array[num_words] int S; // sentence
}

parameters {
  matrix[1 + num_topics, 2] active_gumbel;
}

// `active` (the extracted output, get_pars()) is the only transformed
// parameter. The per-topic weight is a loop local and S_weight / S_logit are
// model-block locals: the same log density, but CmdStan does not write
// 1 + num_topics + 2 * num_words extra columns for every draw (upstream
// PPLBench sampled through PyStan with pars=get_pars(), which wrote only
// `active`).
transformed parameters {
  matrix<lower = 0, upper = 1> [1 + num_topics, 2] active;

  // leak node
  active[1] = [1, 0];

  for (j in 2:(1+num_topics)) {
      // for each topic the weight is the negative log of one minus probability
      // note: that this is not the exact model since we are multiplying
      // the edge_weight with the softmax of active rather than argmax
      real active_weight = edge_weight[j, :(j-1)] * active[:(j-1), 1];
      // note: argmax(log(p) + gumbel, log(1-p) + gumbel)
      // is the same as sampling with Bernoulli(p)
      active[j] = softmax(
          ([log1m_exp(-active_weight), -active_weight]+ active_gumbel[j])' / tau
      )';
  }
}

model {
  vector[num_words] S_weight = edge_weight[(1+num_topics+1):] * active[:, 1];
  // note: if p = 1 - exp(-w), then logit(p)=log(1-exp(-w)) + w
  vector[num_words] S_logit = log1m_exp(-S_weight) + S_weight;

  active_gumbel[:, 1] ~ gumbel(0, 1);
  active_gumbel[:, 2] ~ gumbel(0, 1);
  S ~ bernoulli_logit(S_logit);
}
"""
