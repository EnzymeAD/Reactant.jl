"""
Block-diagonal mass matrix benchmark v2: larger scale, stronger correlations.

3 independent regression groups, K=50 per group, P=150 total.
High collinearity (condition_number=200) for strong posterior correlations.
Dense mass: 150x150 Cholesky per iteration.
Block-diagonal: 3 x 50x50 Cholesky per iteration (27x cheaper).
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.diagnostics import effective_sample_size
import time

K = 50    # params per block
N_GROUPS = 3
N_OBS = 300
P_TOTAL = K * N_GROUPS

def make_collinear_X(key, n, k, condition_number=200.0):
    key1, key2 = jax.random.split(key)
    U, _, _ = jnp.linalg.svd(jax.random.normal(key1, (n, k)), full_matrices=False)
    V, _, _ = jnp.linalg.svd(jax.random.normal(key2, (k, k)), full_matrices=False)
    s = jnp.linspace(1.0, 1.0 / condition_number, k)
    return U * s[None, :] @ V.T

def model(X1, y1, X2, y2, X3, y3):
    k = X1.shape[1]
    beta1 = numpyro.sample('beta1', dist.Normal(jnp.zeros(k), 10.0 * jnp.ones(k)))
    beta2 = numpyro.sample('beta2', dist.Normal(jnp.zeros(k), 10.0 * jnp.ones(k)))
    beta3 = numpyro.sample('beta3', dist.Normal(jnp.zeros(k), 10.0 * jnp.ones(k)))
    numpyro.sample('y1', dist.Normal(X1 @ beta1, 1.0), obs=y1)
    numpyro.sample('y2', dist.Normal(X2 @ beta2, 1.0), obs=y2)
    numpyro.sample('y3', dist.Normal(X3 @ beta3, 1.0), obs=y3)

def run_nuts(label, dense_mass, args, num_warmup=1000, num_samples=1000):
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)

    # Warmup (JIT)
    mcmc.run(jax.random.PRNGKey(0), *args)
    jax.block_until_ready(mcmc.get_samples())

    # Timed run
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(42), *args)
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    elapsed = time.perf_counter() - t0

    all_samples = jnp.concatenate([samples['beta1'], samples['beta2'], samples['beta3']], axis=1)
    ess = effective_sample_size(all_samples[None, :, :])
    min_ess = float(jnp.min(ess))
    mean_ess = float(jnp.mean(ess))

    print(f"{label}:")
    print(f"  Time: {elapsed*1000:.0f} ms  |  Min ESS: {min_ess:.0f}  |  Mean ESS: {mean_ess:.0f}  |  n_eff/s: {mean_ess/elapsed:.0f}")
    return elapsed, mean_ess, mean_ess / elapsed

# Generate data
keys = jax.random.split(jax.random.PRNGKey(0), 10)
X1 = make_collinear_X(keys[0], N_OBS, K)
X2 = make_collinear_X(keys[1], N_OBS, K)
X3 = make_collinear_X(keys[2], N_OBS, K)
y1 = X1 @ jax.random.normal(keys[3], (K,)) + jax.random.normal(keys[4], (N_OBS,))
y2 = X2 @ jax.random.normal(keys[5], (K,)) + jax.random.normal(keys[6], (N_OBS,))
y3 = X3 @ jax.random.normal(keys[7], (K,)) + jax.random.normal(keys[8], (N_OBS,))
args = (X1, y1, X2, y2, X3, y3)

print(f"=== Block Mass Matrix Benchmark ===")
print(f"{N_GROUPS} independent groups, K={K} per group, P={P_TOTAL} total, N={N_OBS} obs/group")
print(f"Dense: {P_TOTAL}x{P_TOTAL} Cholesky vs Block-diag: {N_GROUPS}x {K}x{K} Cholesky")
print(f"Cost ratio (Cholesky): {P_TOTAL**3 / (N_GROUPS * K**3):.1f}x\n")

_, _, r1 = run_nuts("Diagonal", False, args)
_, _, r2 = run_nuts("Dense (full)", True, args)
_, _, r3 = run_nuts("Block-diagonal", [("beta1",), ("beta2",), ("beta3",)], args)

print(f"\n=== Summary (n_eff/s) ===")
print(f"  Diagonal:       {r1:.0f}")
print(f"  Dense:          {r2:.0f}")
print(f"  Block-diagonal: {r3:.0f}")
print(f"  Block/Diag:     {r3/r1:.2f}x")
print(f"  Block/Dense:    {r3/r2:.2f}x")
