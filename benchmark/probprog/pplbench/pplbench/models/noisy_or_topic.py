# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel
from .utils import log1mexpm, split_train_test


LOGGER = logging.getLogger(__name__)


class NoisyOrTopic(BaseModel):
    """
    Noisy-Or Topic Model aka Latent Keyphrase Inference (LAKI)

    Reference paper: http://hanj.cs.illinois.edu/pdf/www16_jliu.pdf

    This is a Bayesian Network with boolean-valued random variables and Noisy-OR
    dependencies. The internal nodes of the network represent "topics" while the
    leaf nodes represent "words". Note that topics can be parents of other topics
    or words and there is a special "leak" node which is the parent of all the nodes.

    These nodes are assigned numbers as follows, node 0 is the special leak node
    followed by `num_topics` topic nodes and finally `num_words` word nodes.

    The leak node is always active (or true) and the inference task is to
    determine the values of the topic nodes given the values of all the word nodes.

    The model consists of a matrix `edge_weight`, s.t. edge_weight[j, k] is the
    edge weight (>=0) from parent node k to child node j. A high edge weight
    suggests that j will likely be true when k is true while a weight of zero
    indicates that there is no direct dependence between j and k.

    We are interested in learning the boolean vector `active` of size 1+num_topics.

    Hyper Parameters:

        n - number of sentences this is fixed to two
        num_topics - number of topics
        num_words - number of words
        avg_fanout - average number of children of a topic
        avg_leak_weight - average weight of a leak-to-node edge
        avg_topic_weight - average weight of a topic-to-node edge

    Model:

        initialize edge_weight[1 + num_topics + num_words, 1 + num_topics] = 0

        for j in 1 .. (num_topics + num_words)
            edge_weight[j, 0] ~ Exp(avg_leak_weight)

        for k in 1 .. num_topics

            draw a subset J from the set {k+1 .. num_topics+num_words}
                s.t. |J| ~ max(1, Poisson(avg_fanout))

            for j in J
                edge_weight[j, k] ~ Exp(avg_topic_weight)

        active[0] = 1

        for k in 1 .. num_topics

            prob_k = 1 - exp( - sum_{l=0}^{k-1} edge_weight[k, l] * active[l] )
            active[k] ~ Bernoulli( prob_k )

        for i in 1 .. n

            for j in 1 .. num_words

                prob_j = 1 - exp( - sum_{l=0}^{num_topics} edge_weight[num_topics+j, l] * active[l] )

                S[i, j] ~ Bernoulli( prob_j )


    The dataset consists of

        S[sentence, word] : {0, 1}  ; sentence=0..n-1,  word=0..num_words-1

    and it includes the attributes

        n                                                          = 1
        num_topics                                                 : integer >= 1
        num_words                                                  : integer >= 1
        edge_weight[1 + num_topics + num_words, 1 + num_topics]    : float

    The posterior samples should include

        active[draw, topic] : {0, 1}    topic=0..num_topics

    note that active[:, 0] should always be 1
    """

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2,
        num_topics: int = 50,
        num_words: int = 100,
        avg_fanout: float = 3.0,
        avg_leak_weight: float = 0.1,
        avg_topic_weight: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        if n != 2:
            raise ValueError("n must be 2 for NoisyOrTopic model")
        if num_topics < 1 or num_words < 1:
            raise ValueError("atleast one topic and one word is required")
        edge_weight = np.zeros((1 + num_topics + num_words, 1 + num_topics))
        active = np.zeros(1 + num_topics)
        rng = np.random.default_rng(seed)
        # generate the leak node and its children edges
        active[0] = 1
        edge_weight[1:, 0] = rng.exponential(
            scale=avg_leak_weight, size=num_topics + num_words
        )
        # generate each of the topic nodes and their children edges
        for node in range(1, num_topics + 1):
            prob = 1 - np.exp(-edge_weight[node, :] @ active)
            active[node] = rng.binomial(1, prob)
            num_children = max(1, rng.poisson(avg_fanout))
            # say we have 4 topics and 7 words and node=2 then the possible children
            # are 3, 4, .. 11
            poss_children = node + 1 + np.arange(num_words + num_topics - node)
            children = rng.choice(
                poss_children, size=min(len(poss_children), num_children), replace=False
            )
            edge_weight[children, node] = rng.exponential(
                avg_topic_weight, len(children)
            )
        # generate the sentences
        wordidx = 1 + num_topics + np.arange(num_words)
        prob = 1 - np.exp(-edge_weight[wordidx] @ active)
        S = rng.binomial(1, prob, size=(n, num_words))

        data = xr.Dataset(
            {"S": (["sentence", "word"], S)},
            coords={"sentence": np.arange(n), "word": np.arange(num_words)},
            attrs={
                "n": n,
                "num_topics": num_topics,
                "num_words": num_words,
                "edge_weight": edge_weight,
            },
        )
        return split_train_test(data, "sentence", 0.5)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Computes the predictive likelihood of all the test items w.r.t. each sample.
        See the class documentation for the `samples` and `test` parameters.
        :returns: a numpy array of the same size as the sample dimension.
        """
        # transpose the datasets to be in a convenient format
        samples = samples.transpose("draw", "topic")
        test = test.transpose("sentence", "word")
        edge_weight = test.edge_weight  # size=(1+num_topics+num_words, 1+num_topics)
        S = test.S[0]  # size=num_words
        active = samples.active.values  # size = (iterations, 1+num_topics)
        if not (active[:, 0] == 1).all():
            raise RuntimeError("leak node should always be active in posterior samples")
        # we will compute the sum of the incoming weight of each word to compute the
        # log likelihood of the data
        wordidx = 1 + test.num_topics + np.arange(test.num_words)
        weight = active @ edge_weight[wordidx].T  # size = (iterations, num_words)
        loglike = np.where(
            S, log1mexpm(weight), -weight
        )  # size = (iterations, num_words)
        return loglike.sum(axis=1)  # size = (iterations,)
