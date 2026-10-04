# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import xarray as xr

from ..base_ppl_impl import MATCHED_INIT_REAL
from .base_stan_impl import BaseStanImplementation


class LogisticRegression(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def matched_init_params(self, data: xr.Dataset) -> Dict:
        """Deterministic canonical init (matched-init control experiment).

        Constrained values identical to the Impulse arm's NUTS start: alpha and
        beta both at MATCHED_INIT_REAL (both real support). Stan inits live on the
        constrained scale, so these are passed straight through as `inits`. Only
        used with --matched-init.

        NOT the origin. Stan's own `bernoulli_logit` is differentiable at 0, so
        this arm takes the offset purely to stay matched with the NumPyro arm,
        which is not (see MATCHED_INIT_REAL in ppls/base_ppl_impl.py). All four
        arms of logistic_regression_matched_init.json must pin the same literal.
        """
        k = int(self.attrs["k"])
        return {
            "alpha": np.full(1, MATCHED_INIT_REAL),
            "beta": np.full(k, MATCHED_INIT_REAL),
        }

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")
        # we already have all the values to be bound except for X and Y in self.attrs
        attrs: dict = self.attrs.copy()
        attrs["X"] = data.X.values
        attrs["Y"] = data.Y.values.astype(int)
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        alpha = np.atleast_1d(samples["alpha"])
        beta = np.atleast_2d(samples["beta"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        if beta.ndim > 2:
            beta = beta.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "beta": (["draw", "feature"], beta),
            },
            coords={
                "draw": np.arange(beta.shape[0]),
                "feature": np.arange(beta.shape[-1]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["alpha", "beta"]

    def get_code(self) -> str:
        """
        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
  int n;
  array[n] int<lower=0, upper=1> Y;
  int k;
  matrix[n, k] X;
  real alpha_scale;
  real beta_scale;
  real beta_loc;
}
parameters {
  real alpha;
  vector[k] beta;
}
model {
  // A model-block local, not a transformed parameter: the same log density,
  // but CmdStan does not write n = 10000 columns of mu for every draw (upstream
  // PPLBench sampled through PyStan with pars=get_pars(), which never wrote mu).
  vector[n] mu = alpha + X * beta;
  alpha ~ normal(0.0, alpha_scale);
  beta ~ normal(beta_loc, beta_scale);
  Y ~ bernoulli_logit(mu);
}
"""
