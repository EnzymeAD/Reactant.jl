import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import genjax
from genjax import ChoiceMap, Regenerate, Selection

@genjax.gen
def linreg(xs, ones_n):
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "slope"
    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(10.0)) @ "intercept"
    ys = genjax.mv_normal_diag(slope * xs + intercept, ones_n) @ "ys"
    return ys

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ys = jnp.array([8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90])
ones_n = jnp.ones(10)
chm = ChoiceMap.d({"ys": ys})
req_s = Regenerate(Selection.at["slope"])
req_i = Regenerate(Selection.at["intercept"])
NUM_ITERS, NUM_STEPS = 500, 1000

def step(i, carry):
    tr, key, acc = carry
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    new_tr, w, _, _ = tr.edit(k1, req_s)
    a1 = jnp.log(jax.random.uniform(k2)) < w
    tr = jax.lax.cond(a1, lambda a, b: a, lambda a, b: b, new_tr, tr)
    new_tr, w, _, _ = tr.edit(k3, req_i)
    a2 = jnp.log(jax.random.uniform(k4)) < w
    tr = jax.lax.cond(a2, lambda a, b: a, lambda a, b: b, new_tr, tr)
    return (tr, key, acc + a1 + a2)

@jax.jit
def kernel(key):
    kg, kl = jax.random.split(key)
    tr, _ = linreg.importance(kg, chm, (xs, ones_n))
    tr, _, acc = jax.lax.fori_loop(0, NUM_ITERS, step, (tr, kl, jnp.array(0)))
    ch = tr.get_choices()
    return ch["slope"], ch["intercept"], acc

key = jax.random.PRNGKey(1)
out = kernel(key); jax.block_until_ready(out)
best = float("inf")
for _ in range(20):
    t0 = time.perf_counter()
    out = kernel(key); jax.block_until_ready(out)
    t = time.perf_counter() - t0
    best = min(best, t)
    print(f"GENJAX linreg REP wall {t:.6f} s | per-move {t/NUM_STEPS*1e6:.4f} us", flush=True)
print(f"GENJAX linreg per-move {best/NUM_STEPS*1e6:.3f} us | accepts {int(out[2])}/1000 | slope={float(out[0]):.4f}")
