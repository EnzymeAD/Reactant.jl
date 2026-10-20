#!/usr/bin/env python
"""
NumPyro reference for the 1D heat-equation Bayesian inverse benchmark (Phase 1).

potential_fn-based NUTS matching the Julia custom logpdf exactly (same
z-parameterization, same full normalization constants). Loads the SAME y data
file written by heat_inverse.jl so both systems consume bit-identical data.

Two forward variants:
  naive   : cholesky factorization INSIDE the lax.scan body
  hoisted : cholesky factorization OUTSIDE the scan (manual LICM)

Matched NUTS settings: num_samples=20, num_warmup=0, max_tree_depth=5,
step_size=0.1, no adaptation, dense identity inverse mass matrix.

Usage:
  export PYTHONPATH="/mnt4/sbrantq/probprog/jax:/mnt4/sbrantq/probprog/numpyro:$PYTHONPATH"
  CUDA_VISIBLE_DEVICES="" /home/sbrantq/anaconda3/envs/probprog/bin/python \
      heat_inverse_numpyro.py <scale> [bench|parity|point]
"""
import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy.linalg as jsl

jax.config.update("jax_enable_x64", True)

import numpyro
from numpyro.infer import NUTS, MCMC

DATA_DIR = os.environ.get("HEATINV_DATA_DIR", "/tmp/sbrantq-heatinv/phase1")
# Match the Julia side (heat_inverse.jl HEATINV_STEP, default 0.02: with 0.1
# every iteration diverges on this tight posterior).
STEP_SIZE = float(os.environ.get("HEATINV_STEP", "0.02"))


def load_data(scale):
    with open(os.path.join(DATA_DIR, f"heat_data_scale{scale}.json")) as f:
        d = json.load(f)
    return dict(
        D=int(d["D"]),
        T=int(d["T"]),
        M_obs=int(d["M_obs"]),
        S=int(d["S"]),
        dt=float(d["dt"]),
        kappa_true=float(d["kappa_true"]),
        sigma_true=float(d["sigma_true"]),
        sensor_idx=np.array(d["sensor_idx"], dtype=np.int64),   # 1-based
        obs_steps=np.array(d["obs_steps"], dtype=np.int64),     # 1-based coarse steps
        y=np.array(d["y"], dtype=np.float64),                   # M_obs x S
        x0=np.array(d["x0_coarse"], dtype=np.float64),          # D
    )


def laplacian(D):
    # c = (D+1)^2 exact in double for the D used here (matches Julia bit-for-bit)
    c = (D + 1.0) ** 2
    L = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        L[i, i] = 2.0 * c
        if i > 0:
            L[i, i - 1] = -c
        if i < D - 1:
            L[i, i + 1] = -c
    return L


def make_potential(data, hoist):
    D, T, dt = data["D"], data["T"], data["dt"]
    L = jnp.array(laplacian(D))
    I_d = jnp.eye(D)
    x0 = jnp.array(data["x0"]).reshape(D, 1)          # D x 1
    y = jnp.array(data["y"])                           # M_obs x S
    obs_idx0 = jnp.array(data["obs_steps"] - 1)        # 0-based rows into traj
    sensor0 = jnp.array(data["sensor_idx"] - 1)        # 0-based sensor columns
    N_obs = data["M_obs"] * data["S"]
    LOG2PI = np.log(2.0 * np.pi)

    def potential(params):
        z = params["z"]
        z_kappa = z[0]
        z_sigma = z[1]
        kappa = jnp.exp(z_kappa)
        sigma = jnp.exp(z_sigma)
        M = I_d + (dt * kappa) * L                      # D x D, SPD

        if hoist:
            cf = jsl.cho_factor(M, lower=False)         # (U, lower=False)

            def body(x, _):
                xn = jsl.cho_solve(cf, x)               # M xn = x
                return xn, xn[:, 0]

        else:

            def body(x, _):
                cf = jsl.cho_factor(M, lower=False)
                xn = jsl.cho_solve(cf, x)
                return xn, xn[:, 0]

        _, traj = lax.scan(body, x0, None, length=T)    # traj: T x D (row k = x_{k+1})
        obs_states = traj[obs_idx0]                      # M_obs x D
        obs_sensors = obs_states[:, sensor0]             # M_obs x S
        resid = y - obs_sensors
        loglik = -0.5 * jnp.sum(resid * resid) / (sigma * sigma) \
            - N_obs * (0.5 * LOG2PI + z_sigma)
        logprior = -0.5 * z_kappa * z_kappa - 0.5 * z_sigma * z_sigma - LOG2PI
        return -(logprior + loglik)                      # potential = -logpdf

    return potential


def rbg_key(seed_u64):
    """Match test/probprog/common.jl seed_to_rbg_key: view uint64[2] as uint32,
    wrap as an rbg key."""
    seed_np = np.array(seed_u64, dtype=np.uint64)
    seed_u32 = seed_np.view(np.uint32)
    return jax.random.wrap_key_data(seed_u32, impl="rbg")


def make_kernel(potential, step_size):
    return NUTS(
        potential_fn=potential,
        step_size=step_size,
        max_tree_depth=5,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        dense_mass=True,
        inverse_mass_matrix=jnp.eye(2),
        find_heuristic_step_size=False,
    )


def run_bench(potential, label, num_samples=20, step_size=STEP_SIZE):
    kernel = make_kernel(potential, step_size)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=num_samples, progress_bar=False)
    init_params = {"z": jnp.array([0.0, 0.0])}
    # Warmup (JIT compile)
    mcmc.run(jax.random.PRNGKey(0), init_params=init_params)
    mcmc.get_samples()
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        mcmc.run(jax.random.PRNGKey(100 + i), init_params=init_params)
        s = mcmc.get_samples()
        jax.block_until_ready(s["z"])
        times.append(time.perf_counter() - t0)
    median_t = sorted(times)[2]
    print(f"{label}:")
    print(f"  Median (20 NUTS samples): {median_t*1000:.1f} ms")
    print(f"  All: {[round(t*1000, 1) for t in times]} ms")
    return median_t


def run_parity(potential, scale, num_samples=20, step_size=STEP_SIZE, seed=(1, 5)):
    kernel = make_kernel(potential, step_size)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=num_samples, progress_bar=False)
    init_params = {"z": jnp.array([0.0, 0.0])}
    mcmc.run(rbg_key(list(seed)), init_params=init_params,
             extra_fields=("diverging",))
    samples = np.asarray(mcmc.get_samples()["z"])   # num_samples x 2
    diverging = np.asarray(mcmc.get_extra_fields().get("diverging", []))
    print("  NumPyro samples (z_kappa, z_sigma):")
    for j in range(samples.shape[0]):
        print(f"    [{j+1}] {samples[j,0]}  {samples[j,1]}")
    out = {
        "seed": list(seed),
        "step_size": step_size,
        "max_tree_depth": 5,
        "samples": [samples[j].tolist() for j in range(samples.shape[0])],
        "diverged": [int(v) for v in diverging] if diverging.size else [],
    }
    with open(os.path.join(DATA_DIR, f"heat_samples_numpyro_scale{scale}.json"), "w") as f:
        json.dump(out, f)
    print(f"  wrote numpyro samples. n_diverged={int(diverging.sum()) if diverging.size else 'n/a'}")
    return samples


def run_point(potential, scale):
    zpts = [[0.0, 0.0], [0.3, -0.2], [-0.5, 0.1], [0.2, 0.5]]
    results = {}
    for z in zpts:
        val = float(-potential({"z": jnp.array(z)}))   # logpdf = -potential
        results[str(z)] = val
        print(f"    logpdf(z={z}) = {val}")
    with open(os.path.join(DATA_DIR, f"heat_logpdf_numpyro_scale{scale}.json"), "w") as f:
        json.dump(results, f)
    print("  wrote numpyro logpdf values.")
    return results


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    mode = sys.argv[2] if len(sys.argv) > 2 else "bench"
    data = load_data(scale)
    print(f"=== NumPyro Heat Inverse: scale {scale}, "
          f"D={data['D']} T={data['T']} M_obs={data['M_obs']} S={data['S']}, mode={mode} ===")

    pot_naive = make_potential(data, hoist=False)
    pot_hoist = make_potential(data, hoist=True)

    if mode == "point":
        # Density values are model-math only (hoist flag irrelevant); use hoisted.
        run_point(pot_hoist, scale)
    elif mode == "parity":
        run_parity(pot_hoist, scale)
    else:
        t_naive = run_bench(pot_naive, "Naive (cholesky inside scan)")
        print()
        t_hoist = run_bench(pot_hoist, "Hoisted (cholesky outside scan)")
        print(f"\nSpeedup (hoisted/naive): {t_naive/t_hoist:.2f}x")


if __name__ == "__main__":
    main()
