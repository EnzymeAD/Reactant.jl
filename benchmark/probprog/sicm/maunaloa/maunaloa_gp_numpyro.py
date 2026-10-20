#!/usr/bin/env python
"""
NumPyro reference for the Mauna Loa CO2 GP-regression SICM benchmark.

potential_fn-based NUTS matching the Julia custom logpdf (maunaloa_gp.jl) exactly:
same z-parameterization (sigma = exp(z), Normal(0, PS) priors on z), same full
MVN normalization, same K_base data file (read bit-identically from the JSON that
maunaloa_gp_prep.py wrote). Purpose: pointwise density parity (1e-12 relative) and
NUTS trajectory lockstep with adaptation off and a fixed step size.

Usage:
  export PYTHONPATH="/mnt4/sbrantq/probprog/jax:/mnt4/sbrantq/probprog/numpyro:$PYTHONPATH"
  CUDA_VISIBLE_DEVICES="" /home/sbrantq/anaconda3/envs/probprog/bin/python \
      maunaloa_gp_numpyro.py <N> [point|parity|bench]
"""
import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

jax.config.update("jax_enable_x64", True)

import numpyro
from numpyro.infer import NUTS, MCMC

DATA_DIR = os.environ.get("MAUNALOA_DATA_DIR", "/tmp/sbrantq-heatinv/maunaloa")
PS = float(os.environ.get("MAUNALOA_PRIOR_SD", "1.0"))
STEP_SIZE = float(os.environ.get("MAUNALOA_STEP", "0.02"))
N_SAMPLES = int(os.environ.get("MAUNALOA_NSAMPLES", "20"))
MAX_TREE_DEPTH = int(os.environ.get("MAUNALOA_MTD", "5"))
LOG2PI = np.log(2.0 * np.pi)


def load_data(N):
    with open(os.path.join(DATA_DIR, f"maunaloa_data_N{N}.json")) as f:
        d = json.load(f)
    n = int(d["attrs"]["n"])
    K = np.array(d["K_base"], dtype=np.float64)
    y = np.array(d["Y"], dtype=np.float64)
    return n, K, y


def make_potential(N, K_base, y):
    Kb = jnp.array(K_base)
    yv = jnp.array(y).reshape(N, 1)
    I_n = jnp.eye(N)

    def potential(params):
        z = params["z"]
        z_f = z[0]
        z_n = z[1]
        sf2 = jnp.exp(2.0 * z_f)
        sn2 = jnp.exp(2.0 * z_n)
        K = sf2 * Kb + sn2 * I_n
        L, low = jsl.cho_factor(K, lower=True)
        sol = jsl.cho_solve((L, low), yv)                  # K^-1 y
        quad = jnp.sum(yv * sol)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))       # log|K|
        loglik = -0.5 * (N * LOG2PI + logdet + quad)
        logprior = -0.5 * (z_f * z_f + z_n * z_n) / (PS * PS) - LOG2PI - 2.0 * np.log(PS)
        return -(logprior + loglik)

    return potential


def rbg_key(seed_u64):
    """Match Reactant ReactantRNG(UInt64[...]): view uint64[2] as uint32, wrap rbg."""
    seed_np = np.array(seed_u64, dtype=np.uint64)
    seed_u32 = seed_np.view(np.uint32)
    return jax.random.wrap_key_data(seed_u32, impl="rbg")


def make_kernel(potential, step_size):
    return NUTS(
        potential_fn=potential,
        step_size=step_size,
        max_tree_depth=MAX_TREE_DEPTH,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        dense_mass=True,
        inverse_mass_matrix=jnp.eye(2),
        find_heuristic_step_size=False,
    )


def run_point(potential, N):
    zpts = [[0.0, 0.0], [0.7, -0.7], [0.3, -0.2], [-0.4, 0.5]]
    results = {}
    for z in zpts:
        val = float(-potential({"z": jnp.array(z)}))
        results[str(z)] = val
        print(f"    logpdf(z={z}) = {val}")
    with open(os.path.join(DATA_DIR, f"maunaloa_logpdf_numpyro_N{N}.json"), "w") as f:
        json.dump(results, f)
    print("  wrote numpyro logpdf values.")
    return results


def run_parity(potential, N, seed=(1, 5)):
    kernel = make_kernel(potential, STEP_SIZE)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=N_SAMPLES, progress_bar=False)
    init_params = {"z": jnp.array([0.0, 0.0])}
    mcmc.run(rbg_key(list(seed)), init_params=init_params, extra_fields=("diverging",))
    samples = np.asarray(mcmc.get_samples()["z"])
    diverging = np.asarray(mcmc.get_extra_fields().get("diverging", []))
    print("  NumPyro samples (z_f, z_n):")
    for j in range(samples.shape[0]):
        print(f"    [{j+1}] {samples[j,0]}  {samples[j,1]}")
    out = {
        "seed": list(seed), "step_size": STEP_SIZE, "max_tree_depth": MAX_TREE_DEPTH,
        "samples": [samples[j].tolist() for j in range(samples.shape[0])],
        "diverged": [int(v) for v in diverging] if diverging.size else [],
    }
    with open(os.path.join(DATA_DIR, f"maunaloa_samples_numpyro_N{N}.json"), "w") as f:
        json.dump(out, f)
    print(f"  wrote numpyro samples. n_diverged={int(diverging.sum()) if diverging.size else 'n/a'}")
    return samples


def run_bench(potential, N):
    kernel = make_kernel(potential, STEP_SIZE)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=N_SAMPLES, progress_bar=False)
    init_params = {"z": jnp.array([0.0, 0.0])}
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
    print(f"NumPyro (cholesky each grad): median {median_t*1000:.1f} ms; "
          f"all {[round(t*1000,1) for t in times]} ms")
    return median_t


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    mode = sys.argv[2] if len(sys.argv) > 2 else "point"
    n, K, y = load_data(N)
    print(f"=== NumPyro Mauna Loa GP: N={n}, step={STEP_SIZE}, mtd={MAX_TREE_DEPTH}, "
          f"samples={N_SAMPLES}, mode={mode} ===")
    potential = make_potential(n, K, y)
    if mode == "point":
        run_point(potential, n)
    elif mode == "parity":
        run_parity(potential, n)
    else:
        run_bench(potential, n)


if __name__ == "__main__":
    main()
