# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class CrowdSourcedAnnotation(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def matched_init_params(self, data: xr.Dataset) -> Dict:
        """Deterministic uniform-simplex init (matched-init control experiment).

        Constrained values identical to the Impulse arm's NUTS start: prev is
        the uniform K-simplex and every confusion row is the uniform K-simplex.
        Stan inits live on the constrained scale, so these are passed straight
        through as `inits`. Only used when the harness runs with --matched-init.
        """
        K = int(self.attrs["num_categories"])
        k = int(self.attrs["k"])
        return {
            "prev": np.full((K,), 1.0 / K),
            "confusion_matrix": np.full((k, K, K), 1.0 / K),
        }

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "item_label")
        attrs: dict = self.attrs.copy()
        # indices in Stan are 1-based
        attrs["labels"] = (data.labels.values + 1).astype(int)
        attrs["labelers"] = (data.labelers.values + 1).astype(int)
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        prev = np.asarray(samples["prev"])
        confusion_matrix = np.asarray(samples["confusion_matrix"])
        # squeeze out the chains dimension if present
        if prev.ndim > 2:
            prev = prev.squeeze(1)
        if confusion_matrix.ndim > 4:
            confusion_matrix = confusion_matrix.squeeze(1)
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

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["prev", "confusion_matrix"]

    def get_code(self) -> str:
        """
        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
    int<lower=1> n;  // number of items
    int<lower=1> k;  // total number of labelers
    int<lower=1> num_categories;
    real<lower=0> expected_correctness;
    real<lower=0> concentration;
    int<lower=1> num_labels_per_item;  // num_labels
    array[n, num_labels_per_item] int<lower=1, upper=num_categories> labels;
    array[n, num_labels_per_item] int<lower=1, upper=k> labelers;
}

transformed data {
  vector[num_categories] beta;
  array[num_categories] vector[num_categories] alpha;
  beta = rep_vector(1./num_categories, num_categories);
  alpha = rep_array(rep_vector(concentration * (1-expected_correctness)
                               / (num_categories-1), num_categories), num_categories);
  for (c in 1:num_categories) {
    alpha[c,c] = concentration * expected_correctness;
  }
}

parameters {
  // prev(Category): The true probabilities of each category.
  simplex[num_categories] prev;
  // confusion_matrix: confusion matrix
  array[k, num_categories] simplex[num_categories] confusion_matrix;
}

model {
  int labeler;
  int label;
  real inner_sum;
  array[num_categories] real log_prob;
  prev ~ dirichlet(beta);
  for (j in 1:k) {
    for (c in 1:num_categories) {
      confusion_matrix[j,c,:] ~ dirichlet(alpha[c,:]);
    }
  }
  for (i in 1:n) {
    for (c in 1:num_categories) {
      inner_sum = 0.0;
      for (j in 1:num_labels_per_item) {
        label = labels[i,j];
        labeler = labelers[i, j];
        inner_sum += categorical_lpmf(label | confusion_matrix[labeler, c]);
      }
      log_prob[c] = inner_sum + log(prev[c]);
    }
    target += log_sum_exp(log_prob);
  }
}
"""
