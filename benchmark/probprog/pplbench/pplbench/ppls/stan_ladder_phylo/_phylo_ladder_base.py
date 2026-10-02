from typing import Dict, List

import numpy as np
import xarray as xr

from ..stan.base_stan_impl import BaseStanImplementation


class PhyloLadderBase(BaseStanImplementation):
    """Shared data plumbing for the phylogenetic Poisson placement variants.

    Every variant receives the tree covariance ``A`` itself (not a factor of
    it), so that where its Cholesky factorization happens is decided by the
    program, not by the harness.
    """

    def __init__(self, n: int, **kwargs) -> None:
        self.n = n

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "N": int(data.attrs["n"]),
            "x": data.X.values.tolist(),
            "y": data.Y.values.astype(int).tolist(),
            "A": data.A.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        def scalar(name):
            v = np.atleast_1d(samples[name])
            return v.squeeze(1) if v.ndim > 1 else v

        sigma_sq = scalar("sigma_sq")
        return xr.Dataset(
            {
                "sigma_sq": (["draw"], sigma_sq),
                "eta": (["draw", "dim"], np.atleast_2d(samples["eta"])),
                "b0": (["draw"], scalar("b0")),
                "b1": (["draw"], scalar("b1")),
                "sigma_obs": (["draw"], scalar("sigma_obs")),
                "eps": (["draw", "dim"], np.atleast_2d(samples["eps"])),
            },
            coords={"draw": np.arange(len(sigma_sq)), "dim": np.arange(self.n)},
        )

    def get_pars(self) -> List[str]:
        return ["sigma_sq", "eta", "b0", "b1", "sigma_obs", "eps"]
