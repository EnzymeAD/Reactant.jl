# Copyright Contributors to the Enzyme project.
# SPDX-License-Identifier: Apache-2.0
"""Fused benchmark model: 1D heat-equation Bayesian inverse problem with a
Gaussian-process prior on the initial condition.

Carries all three redundancy tiers in ONE model:

  TIER 0  data-only, O(D^2)  -- the kernel build K[i,j] = exp(-|s_i-s_j|/rho)
                                from the sensor locations and the fixed
                                lengthscale. Hoistable to `transformed data`;
                                SlicStan finds this one.
  TIER 1  sample-DEPENDENT but source-loop-INVARIANT, O(D^3) -- chol(M(kappa))
                                with M(kappa) = I + dt*kappa*Lap, written inside
                                the implicit-Euler time loop. Standard LICM
                                recovers it before lowering; SlicStan's
                                three-level placement structurally cannot, and
                                post-AD passes cannot either (the unhoisted loop
                                contains a per-iteration Cholesky pullback whose
                                cotangent varies with t).
  TIER 2  algebraic, O(D^3)  -- chol(tau^2 K), which SICM's CholeskyScale
                                factorization splits into tau * chol(K); and
                                M(kappa) = 1*I + (dt*kappa)*Lap, which is
                                CholeskyEigenLift's s*A + t*I target.

Model:
    z_tau, z_kappa ~ Normal(0, 1);  tau = exp(z_tau), kappa = exp(z_kappa)
    u_tilde ~ Normal(0, 1)^D
    u_0 = cholesky(tau^2 K) u_tilde                       (GP prior on the IC)
    M(kappa) u_{t+1} = u_t,  M(kappa) = I + dt*kappa*Lap  (implicit Euler)
    y_j ~ Normal(H u_{j*obs_step}, sigma),  j = 1..M_obs  (S sensors)

Observation-WINDOW form, not every-step: the T steps are split into M_obs windows
of obs_step = T/M_obs steps each, and the likelihood is accumulated BETWEEN
windows. Two reasons, both load-bearing:
  * obs_step must EXCEED Reactant's WHILE_UNROLL_THRESHOLD (4,
    Reactant.jl/src/compiler/OptimizationPasses.jl:35) or the traced time loop is
    unrolled away, the loop-invariant Cholesky has no loop to sit inside, and the
    tier-1 rung collapses (arms A and B become identical).
  * the region/SICM path does not support sample sites INSIDE a traced loop, so
    the likelihood has to live between windows, at trace top level.

Kernel choice: Matern-1/2 (exponential), NOT squared-exponential. The SE kernel
on a D-point unit grid is numerically singular for every physically meaningful
lengthscale once D >= 100 (min eigenvalue goes negative; LAPACK dpotrf fails and
Stan's cholesky_decompose rejects). Matern-1/2 at rho=0.1 gives cond(K) = 3.4e3
at D=300 and 9.4e3 at D=500. `|d|` is written `sqrt(d*d)` so the model stays
inside SlicStan's primitive set (it has `sqrt`, but no `abs`).

Provenance: problem class is standard Bayesian coefficient inversion for the heat
equation (Stuart 2010 Acta Numerica framing; inverse-crime protocol per Kaipio &
Somersalo 2005), with a Matern GP prior on the unknown initial condition. The
INSTANCE (grid, sensors, priors, scales) is constructed for this paper.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from .base_model import BaseModel

LOGGER = logging.getLogger(__name__)


def _matern12(s: np.ndarray, rho: float) -> np.ndarray:
    """Matern-1/2 (exponential) kernel; the spelling the Stan programs use."""
    d = np.abs(s[:, None] - s[None, :])
    return np.exp(-d / rho)


def _laplacian(D: int) -> np.ndarray:
    """Discrete negative Laplacian on [0,1] with Dirichlet ends, scaled 1/h^2.
    c = (D+1)^2 is exact in double, so this is bit-identical across languages."""
    c = (D + 1.0) ** 2
    L = np.diag(np.full(D, 2.0 * c))
    L += np.diag(np.full(D - 1, -c), 1)
    L += np.diag(np.full(D - 1, -c), -1)
    return L


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


class HeatGPFused(BaseModel):
    """Fused heat-inverse + GP-prior model. `n` is the grid size D."""

    @staticmethod
    def generate_init_params(seed: int, **model_attrs) -> Dict[str, np.ndarray]:
        n = int(model_attrs.get("n", 50))
        return {
            "z_tau": np.zeros(1, dtype=np.float64),
            "z_kappa": np.zeros(1, dtype=np.float64),
            "u_tilde": np.zeros(n, dtype=np.float64),
        }

    @staticmethod
    def generate_data(
        seed: int,
        n: int = 100,
        T: int = 20,
        M_obs: int = 4,
        S: int = 8,
        rho: float = 0.1,
        sigma: float = 0.02,
        kappa_true: float = 0.5,
        tau_true: float = 0.3,
        T_final: float = 0.02,
        train_frac: float = 1.0,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        rng = np.random.default_rng(seed)
        D = n
        h = 1.0 / (D + 1)
        s = np.arange(1, D + 1) * h
        Lap = _laplacian(D)
        dt = T_final / T
        assert T % M_obs == 0, f"T ({T}) must be divisible by M_obs ({M_obs})"
        obs_step = T // M_obs
        assert obs_step > 4, (
            f"obs_step ({obs_step}) must exceed WHILE_UNROLL_THRESHOLD (4) or the "
            "traced time loop is unrolled and the tier-1 rung collapses")

        sensor_idx = [int(round(k * (D + 1) / (S + 1))) for k in range(1, S + 1)]
        assert len(set(sensor_idx)) == S, "sensor indices collided"
        H = np.zeros((S, D))
        for a, si in enumerate(sensor_idx):
            H[a, si - 1] = 1.0

        # Inverse-crime avoidance: simulate the truth on a 2x finer grid with
        # dt/2, then restrict to the coarse sensors at the coarse times.
        Df = 2 * D + 1
        sf = np.arange(1, Df + 1) / (Df + 1.0)
        Lapf = _laplacian(Df)
        Kf = _matern12(sf, rho)
        u0f = tau_true * (np.linalg.cholesky(Kf) @ rng.standard_normal(Df))
        Ff = np.linalg.cholesky(np.eye(Df) + (dt / 2 * kappa_true) * Lapf)

        y = np.zeros((M_obs, S))
        xf = u0f.copy()
        fine_sensor = [2 * si - 1 for si in sensor_idx]
        for t in range(1, T + 1):
            xf = _chol_solve(Ff, _chol_solve(Ff, xf))   # two fine = one coarse
            if t % obs_step == 0:
                y[t // obs_step - 1, :] = xf[fine_sensor] + sigma * rng.standard_normal(S)

        data = xr.Dataset(
            {
                "S_loc": (["point"], s),
                "Lap": (["point", "point2"], Lap),
                "H": (["sensor", "point3"], H),
                "Y": (["obs_time", "sensor2"], y),
            },
            coords={
                "point": np.arange(D), "point2": np.arange(D),
                "point3": np.arange(D), "sensor": np.arange(S),
                "sensor2": np.arange(S), "obs_time": np.arange(M_obs),
            },
            attrs={"n": D, "T": T, "M_obs": M_obs, "obs_step": obs_step,
                   "S": S, "rho": rho, "dt": dt,
                   "sigma": sigma, "kappa_true": kappa_true,
                   "tau_true": tau_true, "T_final": T_final},
        )
        return data, data

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        a = test.attrs
        D, T, S = int(a["n"]), int(a["T"]), int(a["S"])
        M_obs, obs_step = int(a["M_obs"]), int(a["obs_step"])
        rho, dt, sigma = float(a["rho"]), float(a["dt"]), float(a["sigma"])
        s = test.S_loc.values
        Lap = test.Lap.values
        H = test.H.values
        y = test.Y.values
        L_K = np.linalg.cholesky(_matern12(s, rho))

        z_tau = np.atleast_1d(samples.z_tau.values)
        z_kappa = np.atleast_1d(samples.z_kappa.values)
        u_tilde = np.atleast_2d(samples.u_tilde.values)
        n_draws = len(z_tau)

        pll = np.zeros(n_draws)
        c = -0.5 * np.log(2.0 * np.pi) - np.log(sigma)
        for i in range(n_draws):
            tau = float(np.exp(np.atleast_1d(z_tau[i]).item()))
            kappa = float(np.exp(np.atleast_1d(z_kappa[i]).item()))
            u = tau * (L_K @ u_tilde[i])
            try:
                LM = np.linalg.cholesky(np.eye(D) + dt * kappa * Lap)
            except np.linalg.LinAlgError:
                pll[i] = -1e10
                continue
            acc = 0.0
            for j in range(M_obs):
                for _ in range(obs_step):
                    u = _chol_solve(LM, u)
                r = y[j] - H @ u
                acc += S * c - 0.5 * float(r @ r) / (sigma * sigma)
            pll[i] = acc
        return pll
