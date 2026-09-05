from typing import Dict, List

import numpy as np
import xarray as xr

from ..stan.base_stan_impl import BaseStanImplementation


class GPLadderBase(BaseStanImplementation):
    """Shared data plumbing for the GP Poisson ladder variants.

    Unlike ``ppls/stan/gp_pois_regr_scaled.py`` (which hands Stan a kernel
    matrix precomputed in Python), every ladder variant receives the raw
    inputs ``x`` and ``rho`` and builds the kernel inside the Stan program, so
    that the placement of that construction is decided by the program (or by
    SlicStan), not by the harness.
    """

    def __init__(self, n: int, rho: float, **kwargs) -> None:
        self.n = n
        self.rho = rho

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        return {
            "N": int(data.attrs["n"]),
            "rho": float(data.attrs["rho"]),
            "x": data.X.values.tolist(),
            "y": data.Y.values.astype(int).tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        alpha = np.atleast_1d(samples["alpha"])
        f_tilde = np.atleast_2d(samples["f_tilde"])
        if alpha.ndim > 1:
            alpha = alpha.squeeze(1)
        return xr.Dataset(
            {
                "alpha": (["draw"], alpha),
                "f_tilde": (["draw", "dim"], f_tilde),
            },
            coords={
                "draw": np.arange(len(alpha)),
                "dim": np.arange(self.n),
            },
        )

    def get_pars(self) -> List[str]:
        return ["alpha", "f_tilde"]
