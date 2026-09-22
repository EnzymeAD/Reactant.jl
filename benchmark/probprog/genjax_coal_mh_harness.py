import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import genjax
from genjax import ChoiceMap, Regenerate, Selection

T = 56
Y = jnp.array([6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0], dtype=jnp.float64)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
ind_pois = genjax.tfp_distribution(lambda lam: tfd.Independent(tfd.Poisson(rate=lam), 1))

@genjax.gen
def coal(T_range, logits):
    s = genjax.categorical(logits) @ "s"
    l1 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l1"
    l2 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l2"
    lam = jnp.where(T_range < s, l1, l2)
    ys = ind_pois(lam) @ "ys"
    return ys

args = (jnp.arange(T), jnp.zeros(T + 1))
chm = ChoiceMap.d({"ys": Y,
                   "l1": jnp.float64(3.0), "l2": jnp.float64(0.9)})
reqs = [Regenerate(Selection.at["s"]), Regenerate(Selection.at["l1"]), Regenerate(Selection.at["l2"])]
NUM_SWEEPS = 2000

def step(i, carry):
    tr, key, hist = carry
    for req in reqs:
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, req)
        a = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a, lambda x, y: x, lambda x, y: y, new_tr, tr)
    s = tr.get_choices()["s"]
    hist = hist.at[s].add(1)
    return (tr, key, hist)

@jax.jit
def kernel(key):
    kg, kl = jax.random.split(key)
    tr, _ = coal.importance(kg, chm, args)
    tr, _, hist = jax.lax.fori_loop(0, NUM_SWEEPS, step, (tr, kl, jnp.zeros(T + 1, dtype=jnp.int64)))
    return hist

key = jax.random.PRNGKey(12345)
hist = kernel(key); jax.block_until_ready(hist)
best = float("inf")
for _ in range(20):
    t0 = time.perf_counter()
    hist = kernel(key); jax.block_until_ready(hist)
    t = time.perf_counter() - t0
    best = min(best, t)
    print(f"GENJAX coal REP wall {t:.6f} s | per-move {t/(3*NUM_SWEEPS)*1e6:.4f} us", flush=True)
hist = jax.device_get(hist)
smap = int(hist.argmax())
mass = float(hist[34:47].sum() / hist.sum())
print(f"GENJAX coal per-move {best/(3*NUM_SWEEPS)*1e6:.3f} us | MAP s = {smap} | mass[34,46] = {mass:.4f}")
