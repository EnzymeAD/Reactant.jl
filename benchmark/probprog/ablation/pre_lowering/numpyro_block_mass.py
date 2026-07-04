"""
Block-diagonal mass matrix benchmark in NumPyro.

Model: Two independent regression problems with collinear predictors.
  Group 1: y1 ~ Normal(X1 @ beta1, 1),  beta1 ~ Normal(0, 10)  (K params)
  Group 2: y2 ~ Normal(X2 @ beta2, 1),  beta2 ~ Normal(0, 10)  (K params)

The two groups are conditionally independent (no shared parameters).
Within each group, collinear design matrices create strong posterior correlations.

Compare three mass matrix configurations:
  - Diagonal: fast but ignores within-block correlations
  - Dense (full 2K x 2K): captures all correlations but O((2K)^3) per iteration
  - Block-diagonal ([beta1], [beta2]): captures within-block correlations at O(2*K^3)

Expected: block-diagonal achieves highest n_eff/s.
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.diagnostics import effective_sample_size
import time
import sys

K = 30   # params per block
N = 200  # observations per group

def make_collinear_X(key, n, k, condition_number=50.0):
    """Create a design matrix with controlled collinearity."""
    key1, key2 = jax.random.split(key)
    # SVD-based: create X with specified condition number
    U, _, _ = jnp.linalg.svd(jax.random.normal(key1, (n, k)), full_matrices=False)
    V, _, _ = jnp.linalg.svd(jax.random.normal(key2, (k, k)), full_matrices=False)
    # Singular values from 1 to 1/condition_number
    s = jnp.linspace(1.0, 1.0 / condition_number, k)
    return U * s[None, :] @ V.T

def model(X1, y1, X2, y2):
    k = X1.shape[1]
    beta1 = numpyro.sample('beta1', dist.Normal(jnp.zeros(k), 10.0 * jnp.ones(k)))
    beta2 = numpyro.sample('beta2', dist.Normal(jnp.zeros(k), 10.0 * jnp.ones(k)))
    numpyro.sample('y1', dist.Normal(X1 @ beta1, 1.0), obs=y1)
    numpyro.sample('y2', dist.Normal(X2 @ beta2, 1.0), obs=y2)

def run_nuts(label, dense_mass, X1, y1, X2, y2, num_warmup=1000, num_samples=1000):
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)

    # Warmup run (JIT compile)
    mcmc.run(jax.random.PRNGKey(0), X1, y1, X2, y2)
    jax.block_until_ready(mcmc.get_samples())

    # Timed run
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(42), X1, y1, X2, y2)
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    elapsed = time.perf_counter() - t0

    # Compute n_eff
    beta1_samples = samples['beta1']  # (num_samples, K)
    beta2_samples = samples['beta2']  # (num_samples, K)
    all_samples = jnp.concatenate([beta1_samples, beta2_samples], axis=1)  # (num_samples, 2K)

    # ESS per parameter, take minimum
    ess_per_param = effective_sample_size(all_samples[None, :, :])  # add chain dim
    min_ess = float(jnp.min(ess_per_param))
    mean_ess = float(jnp.mean(ess_per_param))
    neff_per_s = mean_ess / elapsed

    print(f"{label}:")
    print(f"  dense_mass = {dense_mass}")
    print(f"  Time: {elapsed*1000:.0f} ms ({num_warmup} warmup + {num_samples} samples)")
    print(f"  Min ESS: {min_ess:.0f}, Mean ESS: {mean_ess:.0f}")
    print(f"  Mean n_eff/s: {neff_per_s:.0f}")
    print()
    return elapsed, mean_ess, neff_per_s

# Generate data
key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)
X1 = make_collinear_X(k1, N, K)
X2 = make_collinear_X(k2, N, K)
beta1_true = jax.random.normal(k3, (K,))
beta2_true = jax.random.normal(k4, (K,))
y1 = X1 @ beta1_true + jax.random.normal(jax.random.PRNGKey(10), (N,))
y2 = X2 @ beta2_true + jax.random.normal(jax.random.PRNGKey(11), (N,))

print(f"=== Block-Diagonal Mass Matrix Benchmark ===")
print(f"K={K} params per block, P={2*K} total, N={N} obs per group\n")

# 1. Diagonal mass matrix
_, _, neff_diag = run_nuts("Diagonal", False, X1, y1, X2, y2)

# 2. Full dense mass matrix
_, _, neff_dense = run_nuts("Dense (full)", True, X1, y1, X2, y2)

# 3. Block-diagonal mass matrix
_, _, neff_block = run_nuts("Block-diagonal", [("beta1",), ("beta2",)], X1, y1, X2, y2)

print("=== Summary ===")
print(f"  Diagonal:       {neff_diag:.0f} n_eff/s")
print(f"  Dense (full):   {neff_dense:.0f} n_eff/s")
print(f"  Block-diagonal: {neff_block:.0f} n_eff/s")
