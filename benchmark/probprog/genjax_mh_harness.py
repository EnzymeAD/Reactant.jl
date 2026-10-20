import os, time, json, math
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

@genjax.gen
def cov_model(xs, R):
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "slope"
    tau = genjax.half_normal(jnp.float64(1.0)) @ "tau"
    y = genjax.mv_normal(slope * xs, (tau * tau) * R) @ "y"
    return y

def data_cov(n):
    xs = np.linspace(-1.0, 1.0, n)
    R = np.array([[1.0 if i == j else 0.3 ** abs(i - j) for j in range(n)] for i in range(n)])
    y = -2.0 * xs + 0.15 * np.linspace(0.4, -0.4, n)
    return jnp.array(xs), jnp.array(R), jnp.array(y)

# density validation vs hand-computed joint logpdf
xs, R, y = data_cov(8)
slope0, tau0 = 0.7, 0.9
chm = ChoiceMap.d({"slope": jnp.float64(slope0), "tau": jnp.float64(tau0), "y": y})
score, _ = cov_model.assess(chm, (xs, R))
lp_slope = -0.5 * math.log(2 * math.pi) - math.log(2.0) - slope0**2 / (2 * 4.0)
lp_tau = math.log(2.0) - 0.5 * math.log(2 * math.pi) - tau0**2 / 2
K = tau0**2 * np.array(R)
r = np.array(y) - slope0 * np.array(xs)
sign, logdet = np.linalg.slogdet(K)
lp_y = -0.5 * (len(y) * math.log(2 * math.pi) + logdet + r @ np.linalg.solve(K, r))
hand = lp_slope + lp_tau + lp_y
print(f"assess={float(score):.12f} hand={hand:.12f} |diff|={abs(float(score)-hand):.3e}")

req_slope = Regenerate(Selection.at["slope"])
req_tau = Regenerate(Selection.at["tau"])
NUM_ITERS, NUM_STEPS = 500, 1000

def make_kernel(xs, R, y):
    chm = ChoiceMap.d({"y": y})
    def step(i, carry):
        tr, key, acc = carry
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        new_tr, w, _, _ = tr.edit(k1, req_slope)
        # before 2026-09-29: a1 = jnp.log(jax.random.uniform(k2)) < w
        w = w - (new_tr.project(k1, req_slope.selection) - tr.project(k1, req_slope.selection))
        a1 = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a1, lambda a, b: a, lambda a, b: b, new_tr, tr)
        new_tr, w, _, _ = tr.edit(k3, req_tau)
        # before 2026-09-29: a2 = jnp.log(jax.random.uniform(k4)) < w
        w = w - (new_tr.project(k3, req_tau.selection) - tr.project(k3, req_tau.selection))
        a2 = jnp.log(jax.random.uniform(k4)) < w
        tr = jax.lax.cond(a2, lambda a, b: a, lambda a, b: b, new_tr, tr)
        return (tr, key, acc + a1 + a2)
    @jax.jit
    def kernel(key):
        kg, kl = jax.random.split(key)
        tr, _ = cov_model.importance(kg, chm, (xs, R))
        tr, _, acc = jax.lax.fori_loop(0, NUM_ITERS, step, (tr, kl, jnp.array(0)))
        ch = tr.get_choices()
        return ch["slope"], ch["tau"], acc
    return kernel

results = {}
for n in (8, 32, 64, 128, 256):
    xs, R, y = data_cov(n)
    kernel = make_kernel(xs, R, y)
    key = jax.random.PRNGKey(1)
    out = kernel(key); jax.block_until_ready(out)   # warm/compile
    best = float("inf")
    for _ in range(20):
        t0 = time.perf_counter()
        out = kernel(key); jax.block_until_ready(out)
        t = time.perf_counter() - t0
        best = min(best, t)
        print(f"GENJAX scaledcov n={n} REP wall {t:.6f} s | per-move {t/NUM_STEPS*1e6:.4f} us", flush=True)
    per_move_us = best / NUM_STEPS * 1e6
    results[n] = {"per_move_us": round(per_move_us, 4), "accepts": int(out[2])}
    print(f"n={n:<4} per-move {per_move_us:10.3f} us | accepts {int(out[2])}/1000 | slope={float(out[0]):.4f} tau={float(out[1]):.6f}")

GENJAX_MH_OUT = os.environ.get("GENJAX_MH_OUT", "/mnt4/sbrantq/probprog/oopsla/evidence/programmable_mh")
with open(os.path.join(GENJAX_MH_OUT, "genjax_baseline_sweep.json"), "w") as f:
    json.dump({"genjax": "0.10.3", "jax": jax.__version__, "x64": True, "reps": 8,
               "num_iters": NUM_ITERS, "mh_steps": NUM_STEPS, "sweep": results}, f, indent=1)
print("written genjax_baseline_sweep.json")
