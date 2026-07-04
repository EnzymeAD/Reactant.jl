"""
Full NUTS comparison: NumPyro with naive forward model (K inside loop).
Same settings as Impulse benchmark: D=100, T=100, 20 samples, no warmup,
max_tree_depth=5, step_size=0.1, no adaptation.
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
import time

D = 100
T = 100

key = jax.random.PRNGKey(42)
W = jax.random.normal(key, (D, D)) * 0.01
x0 = jnp.ones((1, D)) * 0.01
I_d = jnp.eye(D)

# Naive logpdf: K(theta) computed INSIDE the loop
def potential_fn_naive(params):
    theta = params['theta']
    def body(t, x):
        K = theta * W.T @ W + I_d
        return x @ K
    x = lax.fori_loop(0, T, body, x0)
    # potential = -logpdf = 0.5 * ||x||^2
    return 0.5 * jnp.sum(x ** 2)

# Manually optimized: K(theta) computed OUTSIDE the loop
def potential_fn_optimized(params):
    theta = params['theta']
    K = theta * W.T @ W + I_d
    def body(t, x):
        return x @ K
    x = lax.fori_loop(0, T, body, x0)
    return 0.5 * jnp.sum(x ** 2)

def run_nuts(potential_fn, label):
    kernel = NUTS(
        potential_fn=potential_fn,
        step_size=0.1,
        max_tree_depth=5,
        adapt_step_size=False,
        adapt_mass_matrix=False,
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=20, progress_bar=False)
    init_params = {'theta': jnp.array(0.5)}

    # Warmup run (JIT compile)
    mcmc.run(jax.random.PRNGKey(0), init_params=init_params)
    mcmc.get_samples()  # force sync

    # Timed runs
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        mcmc.run(jax.random.PRNGKey(100 + i), init_params=init_params)
        samples = mcmc.get_samples()
        # Force sync
        jax.block_until_ready(samples['theta'])
        times.append(time.perf_counter() - t0)

    median_t = sorted(times)[2]
    print(f"{label}:")
    print(f"  Median (20 NUTS samples): {median_t*1000:.1f} ms")
    print(f"  All: {[round(t*1000, 1) for t in times]} ms")
    return median_t

print(f"=== NumPyro NUTS, D={D}, T={T} ===\n")
t_naive = run_nuts(potential_fn_naive, "Naive (K inside loop)")
print()
t_opt = run_nuts(potential_fn_optimized, "Optimized (K outside loop)")
print(f"\nSpeedup (optimized/naive): {t_naive/t_opt:.1f}x")
print(f"\nFor reference: Impulse (auto-optimized) = 174 ms")
