import json
import os
from typing import Dict, List

import numpy as np
import xarray as xr

from ..stan.base_stan_impl import BaseStanImplementation

ULAM_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "real_data", "primates301_m14_10_ulam_standata.json",
)


class PrimatesBrownianStanBase(BaseStanImplementation):
    """Data plumbing shared by the m14.10 Stan variants.

    B, M, G and R come from the PPLBench dataset (the Impulse data file). V and
    Imat are declared but unused by the ulam program; they are passed from the
    ulam data list, whose B, M, G and R are bitwise equal to the dataset's
    (asserted here).
    """

    def __init__(self, n: int, **kwargs) -> None:
        self.n = n

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        X = data.X.values
        B, M, G, R = data.Y.values, X[:, 1], X[:, 2], data.R.values
        with open(ULAM_DATA) as f:
            ulam = json.load(f)
        for name, ours in (("B", B), ("M", M), ("G", G), ("R", R)):
            assert np.array_equal(np.asarray(ulam[name], dtype=np.float64), ours), name
        return {
            "V": ulam["V"],
            "Imat": ulam["Imat"],
            "N_spp": int(data.attrs["n"]),
            "B": B.tolist(),
            "G": G.tolist(),
            "M": M.tolist(),
            "R": R.tolist(),
        }

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        # Called on CmdStan's variables (bM, bG) in the harness and again on the
        # harness output (b) in the driver.
        def scalar(name):
            return np.asarray(samples[name], dtype=np.float64).reshape(-1)

        a = scalar("a")
        if "b" in samples:
            b = np.asarray(samples["b"], dtype=np.float64).reshape(len(a), 2)
        else:
            b = np.stack([scalar("bM"), scalar("bG")], axis=1)
        return xr.Dataset(
            {
                "a": (["draw"], a),
                "b": (["draw", "slope"], b),
                "sigma_sq": (["draw"], scalar("sigma_sq")),
            },
            coords={"draw": np.arange(len(a)), "slope": np.arange(2)},
        )

    def get_pars(self) -> List[str]:
        return ["a", "bG", "bM", "sigma_sq"]
