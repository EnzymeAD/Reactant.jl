# Line vs sine with a sampled period (2026-10-03, evidence
# oopsla/evidence/fig13_stockwait_2026-10-03): the model of Gen's "Introduction
# to Modeling in Gen" samples the sine's period, period ~ gamma(5, 1) (shape 5,
# scale 1; tfd.Gamma(concentration 5, rate 1)), and the sine mean is
# amp * sin(2 pi x / period + phase). Otherwise identical to genjax_linesine_mh_harness.py
# (priors, data, key, projected-score acceptance correction); the MH sweep gains
# a prior-resimulation move on period (six moves; per-move = wall / (6 * sweeps)).
# importance draws every latent, period included, from its prior (only ys is
# constrained). Mean construct: both means computed, selected with jnp.where.
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import genjax
from genjax import ChoiceMap, Regenerate, Selection
# 2026-09-29: MH acceptance fixed. In GenJAX 0.10.3 the weight of a Regenerate
# edit is the joint score difference log p(t') - log p(t), which includes the
# regenerated site's own prior ratio (distribution.py edit_regenerate,
# `incremental_w = w - trace.get_score()`). Used as the acceptance ratio of a
# prior-resimulation proposal it counts that prior twice (the chain targeted
# prior^2 x likelihood). The proposal term log p_sel(v') - log p_sel(v) is now
# subtracted via Trace.project; see
# oopsla/evidence/coal_jarrett_2026-09-29/GENJAX_REGENERATE_WEIGHT.md.

X = jnp.array([0.0, 0.2759, 0.5517, 0.8276, 1.1034, 1.3793, 1.6552, 1.931, 2.2069, 2.4828, 2.7586, 3.0345, 3.3103, 3.5862, 3.8621, 4.1379, 4.4138, 4.6897, 4.9655, 5.2414, 5.5172, 5.7931, 6.069, 6.3448, 6.6207, 6.8966, 7.1724, 7.4483, 7.7241, 8.0])
Y = jnp.array([-0.4453, -0.8217, -0.4477, -0.3201, -0.9794, -0.1994, 0.5287, 0.1443, 1.2569, 1.8079, 1.0433, 1.9606, 2.2105, 3.1492, 2.9465, 3.6189, 2.5726, 3.3957, 3.6323, 4.1488, 5.7783, 4.2277, 5.3055, 5.5581, 5.9808, 5.735, 6.3052, 6.6012, 5.8466, 6.8494])

@genjax.gen
def linesine(xs):
    m = genjax.flip(jnp.float64(0.5)) @ "m"
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "slope"
    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(3.0)) @ "intercept"
    amp = genjax.normal(jnp.float64(0.0), jnp.float64(3.0)) @ "amp"
    phase = genjax.uniform(jnp.float64(0.0), jnp.float64(2 * jnp.pi)) @ "phase"
    period = genjax.gamma(jnp.float64(5.0), jnp.float64(1.0)) @ "period"
    muL = slope * xs + intercept
    muS = amp * jnp.sin(2 * jnp.pi * xs / period + phase)
    mu = jnp.where(m, muL, muS)
    ys = genjax.mv_normal_diag(mu, jnp.full(xs.shape, 0.5)) @ "ys"
    return ys

chm = ChoiceMap.d({"ys": Y})
reqs = [Regenerate(Selection.at[a]) for a in ("m", "slope", "intercept", "amp", "phase", "period")]
NUM_SWEEPS = 1000
MOVES = NUM_SWEEPS * 6

def step(i, carry):
    tr, key, macc = carry
    for req in reqs:
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, req)
        # before 2026-09-29: a = jnp.log(jax.random.uniform(k2)) < w
        w = w - (new_tr.project(k1, req.selection) - tr.project(k1, req.selection))
        a = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a, lambda x, y: x, lambda x, y: y, new_tr, tr)
    return (tr, key, macc + tr.get_choices()["m"])

@jax.jit
def kernel(key):
    kg, kl = jax.random.split(key)
    tr, _ = linesine.importance(kg, chm, (X,))
    tr, _, macc = jax.lax.fori_loop(0, NUM_SWEEPS, step, (tr, kl, jnp.array(0)))
    return macc

key = jax.random.PRNGKey(4242)
out = kernel(key); jax.block_until_ready(out)
best = float("inf")
for _ in range(20):
    t0 = time.perf_counter()
    out = kernel(key); jax.block_until_ready(out)
    t = time.perf_counter() - t0
    best = min(best, t)
    print(f"GENJAX linesine_period REP wall {t:.6f} s | per-move {t/MOVES*1e6:.4f} us", flush=True)
print(f"GENJAX linesine_period per-move {best/MOVES*1e6:.3f} us | P(m=line) = {int(out)/NUM_SWEEPS:.4f}")
