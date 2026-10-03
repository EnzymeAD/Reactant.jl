import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
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

import json, os
# COAL_DATA selects the series:
#   simulated (default)  T = 56 simulated counts, chain warm-started at the
#                        simulation's true rates (l1 = 3.0, l2 = 0.9)
#   jarrett              the recorded British series 1851..1961 (T = 111, the
#                        two missing years dropped, every other year at its
#                        calendar index), chain started from the prior (only
#                        the counts constrained). See
#                        pplbench/pplbench/models/real_data/README.md.
COAL_DATA = os.environ.get("COAL_DATA", "simulated")
if COAL_DATA == "jarrett":
    _raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "pplbench", "pplbench", "models", "real_data",
                                       "coal_mining_jarrett.json")))
    T = len(_raw["counts"])
    TIDX = jnp.array([i for i, c in enumerate(_raw["counts"]) if c is not None])
    Y = jnp.array([c for c in _raw["counts"] if c is not None], dtype=jnp.float64)
else:
    assert COAL_DATA == "simulated", COAL_DATA
    T = 56
    Y = jnp.array([6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0], dtype=jnp.float64)
    TIDX = jnp.arange(T)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
ind_pois = genjax.tfp_distribution(lambda lam: tfd.Independent(tfd.Poisson(rate=lam), 1))

# GENJAX_COAL_LOGITS (2026-10-03, figure 13 round 2):
#   arg    (default) the uniform switch prior's logits are a model argument, as
#          through the 2026-10-03 single window; they are carried in the trace,
#          so every move on s recomputes their logsumexp (112 exps and a log)
#   const  the logits are a constant written in the model, the form a GenJAX user
#          writes for a fixed prior; XLA folds the logsumexp, so the per-move work
#          matches Impulse's DiscreteUniform and Gen.jl's uniform_discrete
#          (constant log density). Same distribution, same draws.
COAL_LOGITS = os.environ.get("GENJAX_COAL_LOGITS", "arg")
assert COAL_LOGITS in ("arg", "const"), COAL_LOGITS
LABEL, TAG = ("GENJAX coal", "genjax") if COAL_LOGITS == "arg" else ("GENJAX coal_constlogits", "genjax-constlogits")

if COAL_LOGITS == "arg":
    @genjax.gen
    def coal(T_range, logits):
        s = genjax.categorical(logits) @ "s"
        l1 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l1"
        l2 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l2"
        lam = jnp.where(T_range < s, l1, l2)
        ys = ind_pois(lam) @ "ys"
        return ys

    args = (TIDX, jnp.zeros(T + 1))
else:
    @genjax.gen
    def coal(T_range):
        s = genjax.categorical(jnp.zeros(T + 1)) @ "s"
        l1 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l1"
        l2 = genjax.gamma(jnp.float64(3.0), jnp.float64(1.0)) @ "l2"
        lam = jnp.where(T_range < s, l1, l2)
        ys = ind_pois(lam) @ "ys"
        return ys

    args = (TIDX,)
if COAL_DATA == "jarrett":
    chm = ChoiceMap.d({"ys": Y})
else:
    chm = ChoiceMap.d({"ys": Y,
                       "l1": jnp.float64(3.0), "l2": jnp.float64(0.9)})
reqs = [Regenerate(Selection.at["s"]), Regenerate(Selection.at["l1"]), Regenerate(Selection.at["l2"])]
NUM_SWEEPS = 2000

def step(i, carry):
    tr, key, hist = carry
    for req in reqs:
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, req)
        # before 2026-09-29: a = jnp.log(jax.random.uniform(k2)) < w
        w = w - (new_tr.project(k1, req.selection) - tr.project(k1, req.selection))
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
    print(f"{LABEL} REP wall {t:.6f} s | per-move {t/(3*NUM_SWEEPS)*1e6:.4f} us", flush=True)
hist = jax.device_get(hist)
smap = int(hist.argmax())
mass = float(hist[34:47].sum() / hist.sum())
print(f"{LABEL} per-move {best/(3*NUM_SWEEPS)*1e6:.3f} us | MAP s = {smap} | mass[34,46] = {mass:.4f}")

if COAL_DATA == "jarrett":
    # Validation (untimed): the timed histogram, then one longer chain that
    # also accumulates the two rates.
    print(f"COAL-HIST {TAG} timed " + " ".join(str(int(v)) for v in hist), flush=True)
    NUM_VAL = int(os.environ.get("COAL_VAL_SWEEPS", "50000"))

    def step_stats(i, carry):
        tr, key, h, m = carry
        tr, key, h = step(i, (tr, key, h))
        ch = tr.get_choices()
        return (tr, key, h, m + jnp.array([ch["l1"], ch["l2"]]))

    @jax.jit
    def kernel_stats(key):
        kg, kl = jax.random.split(key)
        tr, _ = coal.importance(kg, chm, args)
        tr, _, h, m = jax.lax.fori_loop(0, NUM_VAL, step_stats,
                                        (tr, kl, jnp.zeros(T + 1, dtype=jnp.int64), jnp.zeros(2)))
        return h, m / NUM_VAL

    vh, vm = jax.device_get(kernel_stats(jax.random.PRNGKey(54321)))
    print(f"COAL-HIST {TAG} val " + " ".join(str(int(v)) for v in vh), flush=True)
    print(f"COAL-VAL {TAG} sweeps={NUM_VAL} E[l1]={vm[0]:.4f} E[l2]={vm[1]:.4f} "
          f"MAP year={1851 + int(vh.argmax())}", flush=True)
