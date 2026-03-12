# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        # dim 1 is the chains dimension so we squeeze it out
        active = samples["active"].squeeze(1)  # shape: [iterations, num_topics, 2]
        active = np.argmax(active, axis=2) == 0  # shape: [iterations, num_topics]
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
  int S[num_words]; // sentence
}

parameters {
  matrix[1 + num_topics, 2] active_gumbel;
}

transformed parameters {
  // for each topic the weight is the negative log of one minus probability
  vector<lower = 0> [1 + num_topics] active_weight;
  matrix<lower = 0, upper = 1> [1 + num_topics, 2] active;
  vector<lower = 0> [num_words] S_weight;
  vector [num_words] S_logit;

  // leak node
  active[1] = [1, 0];
  active_weight[1] = 0; //unused but needs to be initialized

  for (j in 2:(1+num_topics)) {
      // note: that this is not the exact model since we are multiplying
      // the edge_weight with the softmax of active rather than argmax
      active_weight[j] = edge_weight[j, :(j-1)] * active[:(j-1), 1];
      // note: argmax(log(p) + gumbel, log(1-p) + gumbel)
      // is the same as sampling with Bernoulli(p)
      active[j] = softmax(
          ([log1m_exp(-active_weight[j]), -active_weight[j]]+ active_gumbel[j])' / tau
      )';
  }

  S_weight = edge_weight[(1+num_topics+1):] * active[:, 1];

  // note: if p = 1 - exp(-w), then logit(p)=log(1-exp(-w)) + w
  S_logit = log1m_exp(-S_weight) + S_weight;
}

model {
  active_gumbel[:, 1] ~ gumbel(0, 1);
  active_gumbel[:, 2] ~ gumbel(0, 1);
  S ~ bernoulli_logit(S_logit);
}
"""
