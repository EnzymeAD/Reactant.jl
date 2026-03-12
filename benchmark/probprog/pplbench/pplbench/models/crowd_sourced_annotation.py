# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import xarray as xr
from scipy.special import logsumexp

from .base_model import BaseModel
from .utils import split_train_test


class CrowdSourcedAnnotation(BaseModel):
    """
    Crowd-Sourced Annotation Model

    Paper describing this: https://www.aclweb.org/anthology/Q14-1025

    This model attempts to find the true label of an item based on labels
    provided by a set of imperfect labelers.

    Hyper Parameters:

        n - total number of items
        k - total number of labelers
        num_categories - number of label classes
        expected_correctness - prior belief on correctness of labelers
        num_labels_per_item - number of labels per item
        concentration - strength of prior on expected_correctness

    Model:
        for c in 0 .. num_categories:
            for c' in 0 .. num_categories:
                alpha[c][c'] = concentration * (
                    expected_correctness for c == c',
                    (1-expected_correctness)/(num_categories-1) for c != c'
                )

        for l in 0 .. k - 1:
            for c in 0 .. num_categories - 1:
                confusion_matrix[l, c] ~ Dirichlet(alpha[c]) # confusion matrix of labelers

        for c in 0 .. num_categories:
            beta[c] = 1/num_categories

        prev ~ Dirichlet(beta) # prevalence of categories

        for i in 0 .. n - 1:
            true_labels[i] ~ Categorical(prev) # true label of item i

        for i in 0 .. n - 1:
            labelers[i] = |num_labels_per_item| labelers chosen at random from k without replacement

        for i in 0 .. n - 1:
            for idx, l in enumerate(labelers[i]):
                labels[i][idx] ~ Categorical(confusion_matrix[l, true_labels[i]])

    The dataset consists of the following:
        labelers [n, num_labels_per_item] - 0 .. k - 1
        labels [n, num_labels_per_item] - 0 .. num_categories - 1

    and it includes the attributes
        n - number of items
        k - total number of labelers
        num_categories - number of label classes
        expected_correctness - prior belief on correctness of labelers
        num_labels_per_item - number of labelers per item
        concentration - strength of prior on expected_correctness

    The posterior samples include the following
        prev[draw, num_categories] - float
        confusion_matrix[draw, labelers, true_category, observed_category] - float
            Note that:
                    prev[draw] - simplex
                    confusion_matrix[draw, labelers, true_category] - simplex


    """

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2000,
        k: int = 10,
        num_categories: int = 3,
        expected_correctness: float = 0.8,
        num_labels_per_item: int = 3,
        concentration: float = 10,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        See the class documentation for an explanation of the other parameters.
        :param train_frac: fraction of data to be used for training (default 0.5)
        """
        rng = np.random.default_rng(seed)
        # choose a true class z for each item
        beta = 1 / num_categories * np.ones(num_categories)
        prev = rng.dirichlet(beta)  # shape [num_categories]
        true_labels = rng.choice(num_categories, p=prev, size=n)  # shape [n]
        # set prior that each labeler on average has 'expected_correctness' chance of getting true label
        alpha = ((1 - expected_correctness) / (num_categories - 1)) * np.ones(
            [num_categories, num_categories]
        )
        np.fill_diagonal(alpha, expected_correctness)
        alpha *= concentration
        # sample confusion matrices for labelers from this dirichlet prior
        confusion_matrix = np.zeros([k, num_categories, num_categories])
        for c in range(num_categories):
            # theta_lc ~ Dirichlet(alpha[c])
            confusion_matrix[:, c] = rng.dirichlet(alpha[c], size=k)
        # select labelers for each item, get their labels for that item
        labelers = np.zeros((n, num_labels_per_item), dtype=np.int32)
        labels = np.zeros((n, num_labels_per_item), dtype=np.int32)
        for i in range(n):
            labelers[i] = rng.choice(k, size=num_labels_per_item, replace=False)
            for idx, lb in enumerate(labelers[i]):
                labels[i, idx] = rng.choice(
                    num_categories, p=confusion_matrix[lb, true_labels[i]]
                )
        data = xr.Dataset(
            {
                "labelers": (["item", "item_label"], labelers),
                "labels": (["item", "item_label"], labels),
            },
            coords={"item": np.arange(n), "item_label": np.arange(num_labels_per_item)},
            attrs={
                "n": n,
                "k": k,
                "num_categories": num_categories,
                "expected_correctness": expected_correctness,
                "num_labels_per_item": num_labels_per_item,
                "concentration": concentration,
            },
        )

        return split_train_test(data, "item", train_frac)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Computes the predictive likelihood of all the test items w.r.t. each sample.
        See the class documentation for the `samples` and `test` parameters.
        :returns: a numpy array of the same size as the sample dimension.

        log_prob_i = 0
        for i in items:
            log_prob_c = []
            for c in num_categories:
                P_c = 0
                for j in num_labels_per_item:
                    P_c += log(P(y_test[i, j]| confusion_matrix*[labelers[i, j], c]))
                log_prob_c[c] = P_c + log(prev*[c])
            log_prob_i += logsumexp(log_prob_c)
        """
        # size: [iterations, num_categories]
        prev = samples.prev.values
        # size: [iterations, k, num_categories, num_categories]
        confusion_matrix = samples.confusion_matrix.values
        labels = test.labels.values
        labelers = test.labelers.values
        # size of confusion_matrix[:, labelers, :, labels]
        # is [n/2, num_categories, iterations, num_categories]
        likelihood = logsumexp(
            np.log(confusion_matrix[:, labelers, :, labels]).sum(axis=1) + np.log(prev),
            axis=2,
        ).sum(axis=0)

        return np.array(likelihood)
