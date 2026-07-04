"""
Compare gradient cost of a discretized linear forward model in JAX.

Two versions:
  A) Naive: K(θ) = θ·W'W + I computed INSIDE the loop (what a user writes)
  B) Manual: K(θ) computed OUTSIDE the loop (what an expert would optimize)

JAX cannot automatically transform A→B because AD happens before XLA's LICM.
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
import time

D = 100
T = 100

# Fixed data
key = jax.random.PRNGKey(42)
W = jax.random.normal(key, (D, D)) * 0.01
x0 = jnp.ones((1, D)) * 0.01
I_d = jnp.eye(D)

# Version A: naive (K inside loop — what a user naturally writes)
def logpdf_naive(theta):
    def body(t, x):
        K = theta * W.T @ W + I_d
        return x @ K
    x = lax.fori_loop(0, T, body, x0)
    return -0.5 * jnp.sum(x ** 2)

# Version B: manually optimized (K outside loop — expert optimization)
def logpdf_optimized(theta):
    K = theta * W.T @ W + I_d
    def body(t, x):
        return x @ K
    x = lax.fori_loop(0, T, body, x0)
    return -0.5 * jnp.sum(x ** 2)

grad_naive = jax.jit(jax.grad(logpdf_naive))
grad_opt = jax.jit(jax.grad(logpdf_optimized))

theta = 0.5

# Warmup (JIT compile)
grad_naive(theta).block_until_ready()
grad_opt(theta).block_until_ready()

# Verify they produce the same gradient
g_naive = float(grad_naive(theta))
g_opt = float(grad_opt(theta))
print(f"Gradient (naive):     {g_naive:.10e}")
print(f"Gradient (optimized): {g_opt:.10e}")
print(f"Match: {abs(g_naive - g_opt) < 1e-6 * abs(g_opt)}")

# Benchmark: single gradient call
N_REPS = 100

# Naive
grad_naive(theta).block_until_ready()  # extra warmup
t0 = time.perf_counter()
for _ in range(N_REPS):
    grad_naive(theta).block_until_ready()
t_naive = (time.perf_counter() - t0) / N_REPS

# Optimized
grad_opt(theta).block_until_ready()  # extra warmup
t0 = time.perf_counter()
for _ in range(N_REPS):
    grad_opt(theta).block_until_ready()
t_opt = (time.perf_counter() - t0) / N_REPS

print(f"\nNaive gradient:     {t_naive*1000:.3f} ms")
print(f"Optimized gradient: {t_opt*1000:.3f} ms")
print(f"Speedup: {t_naive/t_opt:.1f}x")

# Also dump HLO to see the structural difference
print("\n=== Naive HLO stats ===")
hlo_naive = jax.jit(jax.grad(logpdf_naive)).lower(theta).compile().as_text()
print(f"  Lines: {hlo_naive.count(chr(10))}")
print(f"  dot: {hlo_naive.count('dot(')}")
print(f"  while: {hlo_naive.count('while(')}")

print("\n=== Optimized HLO stats ===")
hlo_opt = jax.jit(jax.grad(logpdf_optimized)).lower(theta).compile().as_text()
print(f"  Lines: {hlo_opt.count(chr(10))}")
print(f"  dot: {hlo_opt.count('dot(')}")
print(f"  while: {hlo_opt.count('while(')}")
