# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from jax.scipy.special import logsumexp

from .base_numpyro_impl import BaseNumPyroImplementation


class CoalMining(BaseNumPyroImplementation):
    def __init__(
        self,
        n: int,
        s_true: int,
        l1_true: float,
        l2_true: float,
    ) -> None:
        self.n = n
        self.s_true = s_true
        self.l1_true = l1_true
        self.l2_true = l2_true

    def matched_init_params(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """Deterministic canonical init (matched-init protocol).

        Constrained point identical to the Impulse arm's NUTS start
        (standard/coal_mining.jl `build_constraint`): l1 = l2 = 1, the two
        positive-support Gamma sites at the image of unconstrained zero under
        the exp transform. Both sites are scalar in this model. Routed through
        init_to_value by numpyro_harness.py; only used with --matched-init.
        """
        return {"l1": np.array(1.0), "l2": np.array(1.0)}

    def model(self, data: xr.Dataset):
        y = jnp.asarray(data.y.values)  # (T,)
        T = y.shape[0]
        l1 = numpyro.sample("l1", dist.Gamma(3.0, 1.0))
        l2 = numpyro.sample("l2", dist.Gamma(3.0, 1.0))
        # Discrete switch s marginalized via prefix/suffix cumsums.
        a = dist.Poisson(l1).log_prob(y)  # (T,)
        b = dist.Poisson(l2).log_prob(y)  # (T,)
        A = jnp.concatenate([jnp.zeros(1), jnp.cumsum(a)])           # (T+1,)
        B = jnp.concatenate([jnp.cumsum(b[::-1])[::-1], jnp.zeros(1)])  # (T+1,)
        lp = -jnp.log(T + 1.0) + A + B                              # (T+1,)
        numpyro.factor("y", logsumexp(lp))

    def extract_data_from_numpyro(
        self, samples: Dict[str, jnp.ndarray]
    ) -> xr.Dataset:
        l1 = np.asarray(samples["l1"]).reshape(-1)
        l2 = np.asarray(samples["l2"]).reshape(-1)
        return xr.Dataset(
            {"l1": (["draw"], l1), "l2": (["draw"], l2)},
            coords={"draw": np.arange(l1.shape[0])},
        )
