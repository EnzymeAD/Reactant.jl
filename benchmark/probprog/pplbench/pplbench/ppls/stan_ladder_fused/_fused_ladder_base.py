from typing import Dict, List

import numpy as np
import xarray as xr

from ..stan.base_stan_impl import BaseStanImplementation


class FusedLadderBase(BaseStanImplementation):
    """Shared data plumbing for the fused heat-inverse + GP-prior ladder.

    Every variant receives the raw inputs (sensor locations ``s``, the fixed
    discrete Laplacian ``Lap``, the observation matrix ``H``, the lengthscale
    ``rho``) and builds both the kernel and every factorization inside the Stan
    program, so that their placement is decided by the program (or by SlicStan),
    not by the harness.
    """

    def __init__(self, n: int, T: int, M_obs: int = 4, S: int = 8,
                 rho: float = 0.1, **kwargs) -> None:
        self.n = n
        self.T = T
        self.M_obs = M_obs
        self.S = S
        self.rho = rho

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        a = data.attrs
        return {
            "D": int(a["n"]),
            "M_obs": int(a["M_obs"]),
            "obs_step": int(a["obs_step"]),
            "S": int(a["S"]),
            "rho": float(a["rho"]),
            "dt": float(a["dt"]),
            "sigma": float(a["sigma"]),
            "s": data.S_loc.values.tolist(),
            "Lap": data.Lap.values.tolist(),
            "Idm": np.eye(int(a["n"])).tolist(),
            "H": data.H.values.tolist(),
            "y": data.Y.values.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        z_tau = np.atleast_1d(samples["z_tau"])
        z_kappa = np.atleast_1d(samples["z_kappa"])
        u_tilde = np.atleast_2d(samples["u_tilde"])
        if z_tau.ndim > 1:
            z_tau = z_tau.squeeze(1)
        if z_kappa.ndim > 1:
            z_kappa = z_kappa.squeeze(1)
        return xr.Dataset(
            {
                "z_tau": (["draw"], z_tau),
                "z_kappa": (["draw"], z_kappa),
                "u_tilde": (["draw", "dim"], u_tilde),
            },
            coords={"draw": np.arange(len(z_tau)), "dim": np.arange(self.n)},
        )

    def get_pars(self) -> List[str]:
        return ["z_tau", "z_kappa", "u_tilde"]
