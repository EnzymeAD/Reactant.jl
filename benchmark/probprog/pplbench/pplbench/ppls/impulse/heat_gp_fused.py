from typing import Dict

import numpy as np
import xarray as xr

from .base_impulse_impl import BaseImpulseImplementation


class HeatGPFused(BaseImpulseImplementation):
    """Impulse arm of benchmark B (fused heat inverse + GP prior).

    The Julia model returns a DIAGONAL inverse mass matrix (a length-P vector)
    from `setup`, matching the Stan rungs' `metric: diag_e`.
    """

    def __init__(self, n: int, T: int, M_obs: int = 4, S: int = 8,
                 rho: float = 0.1, **kwargs) -> None:
        self.n = n
        self.T = T
        self.M_obs = M_obs
        self.S = S
        self.rho = rho

    @property
    def julia_model_path(self) -> str:
        return "sicm/heat_gp_fused"

    def extract_data_from_impulse(self, samples: Dict[str, np.ndarray]) -> xr.Dataset:
        z_tau = np.asarray(samples["z_tau"]).reshape(-1)
        z_kappa = np.asarray(samples["z_kappa"]).reshape(-1)
        u_tilde = np.atleast_2d(np.asarray(samples["u_tilde"]))
        return xr.Dataset(
            {
                "z_tau": (["draw"], z_tau),
                "z_kappa": (["draw"], z_kappa),
                "u_tilde": (["draw", "dim"], u_tilde),
            },
            coords={"draw": np.arange(len(z_tau)), "dim": np.arange(self.n)},
        )
