# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

# Canonical matched-init value for a REAL-support site whose value feeds a
# reference log-prob that is non-differentiable at the origin.
#
# WHY THIS IS NOT ZERO. The matched-init protocol starts every arm from the
# canonical point of the model's support (real -> 0, positive -> 1, simplex ->
# uniform). For most models that point is fine. But numpyro's
# `Bernoulli(logits=x).log_prob(y)` routes through
#   binary_cross_entropy_with_logits(x, y) = clip(x, 0) + log1p(exp(-|x|)) - x*y
# (numpyro/distributions/util.py), a sum of two pieces that are each
# non-differentiable at x = 0. JAX resolves them with INCONSISTENT conventions:
# clip'(0) = 0.5 (centred) while |x|'(0) = +1 (one-sided). The two kinks
# therefore do not cancel, and the reported gradient of log p is `y` instead of
# the true `y - sigmoid(0) = y - 0.5`: an error of exactly 0.5 per observation.
# Either one-sided convention would have been self-consistent and correct; it is
# the mixture that is wrong. The underlying density `y*x - log(1 + exp(x))` is
# smooth at x = 0, so this is purely an artifact of the reference
# implementation's algebraic form, not a property of the model.
#
# A logit that is a LINEAR map of zero-initialised real-support sites is then
# EXACTLY 0 for every observation, so the canonical init lands the whole
# likelihood on that artifact. `--validate` would compare Impulse's correct
# trajectory against a NumPyro trajectory built from a wrong gradient and blame
# Impulse. Offsetting the real-support sites by this much leaves the
# measure-zero artifact while changing nothing that matters: on
# logistic_regression it moves U by 2.26e-2 out of 3487 (relative 6.5e-6) and
# leaves lambda_max at 5.559311e+06 to nine digits, so the eps_crit analysis
# behind main.py's `_VALIDATE_STEP_SIZE` table is unaffected. Every logit ends up
# at least 5.2e-9 away from the kink.
#
# Apply this ONLY to sites with real support, where the constrained and
# unconstrained values coincide, so the offset needs no transform and every arm
# can pin the identical literal. Positive/simplex sites keep their canonical
# constrained value (1, uniform) and their exact transform (log(1) = 0).
#
# EVERY ARM MUST CARRY THE SAME LITERAL, or the matched-init protocol stops
# comparing a shared starting point. The four homes are:
#   NumPyro / Stan   this constant
#   Impulse          harness.jl `MATCHED_INIT_REAL`
#   Turing           turing_harness.jl `MATCHED_INIT_REAL`
# Only NumPyro actually suffers the kink; the others take the offset purely to
# stay matched.
#
# Models needing it today: logistic_regression (alpha, beta). Latent hazard:
# gp_classify, whose `f = L @ f_tilde` is likewise linear in a real site, so f is
# exactly 0 at f_tilde = 0; it has no `matched_init_params` on any arm yet, and
# whoever adds one must use this constant everywhere.
MATCHED_INIT_REAL = 1e-6


class BasePPLImplementation(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the model representation in the PPL.
        """
        pass
