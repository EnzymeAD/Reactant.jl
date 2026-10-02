"""GenJAX cell of the Gen-tutorial line-model row of the generality figure.

Model: `line_model` of Gen's "Introduction to Modeling in Gen"
(slope ~ normal(0, 1), intercept ~ normal(0, 2), y_i ~ normal(slope*x_i +
intercept, 0.1)). Kernel: the block prior-resimulation move of "Iterative
Inference in Gen", regenerate {slope, intercept} once per sweep, 500 sweeps,
after importance with the ys observed. Timed span: importance + loop in one
jitted kernel, 20 repetitions after one discarded warm-up; per-move = wall/500.

Encoding: one vector site ys ~ mv_normal_diag (as genjax_mh_linreg_harness.py).
The per-point counterpart of the tutorial's (:y, i) loop (genjax.normal
vmapped over the points, encoding "vmap") cannot run this kernel in GenJAX
0.10.3: Vmap.edit implements only Update and IndexRequest and raises
NotImplementedError for the Regenerate request (vmap.py, Vmap.edit). It is kept
selectable (GENJAX_LT_ENCODINGS=vmap) to document that.

MH acceptance: in GenJAX 0.10.3 the weight of a Regenerate edit is the joint
score difference, which includes the regenerated sites' own prior ratio; the
proposal term is subtracted via Trace.project, exactly as in
genjax_coal_mh_harness.py (oopsla/evidence/coal_jarrett_2026-09-29/
GENJAX_REGENERATE_WEIGHT.md).

After the timed repetitions (untimed): assess() of a fixed choice map against
the hand-computed joint log density, acceptance count, final slope/intercept.

Usage: python genjax_mh_linreg_tutorial_harness.py <data.json> <A|B>
"""
import json, os, sys, time, statistics
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import genjax
from genjax import ChoiceMap, Regenerate, Selection

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "linreg_tutorial_data.json")
ROW = sys.argv[2] if len(sys.argv) > 2 else "A"
ENCODINGS = os.environ.get("GENJAX_LT_ENCODINGS", "vec").split()
NUM_SWEEPS, NREPS = 500, 20
d = json.load(open(DATA))[ROW]
XS = jnp.array(d["xs"], dtype=jnp.float64)
YS = jnp.array(d["ys"], dtype=jnp.float64)
N = XS.shape[0]
SIG = jnp.full(N, 0.1)


@genjax.gen
def line_vec(xs, sig):
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(1.0)) @ "slope"
    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "intercept"
    ys = genjax.mv_normal_diag(slope * xs + intercept, sig) @ "ys"
    return ys


@genjax.gen
def line_vmap(xs, sig):
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(1.0)) @ "slope"
    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "intercept"
    ys = genjax.normal.vmap(in_axes=(0, 0))(slope * xs + intercept, sig) @ "ys"
    return ys


MODELS = {"vec": line_vec, "vmap": line_vmap}
REQ = Regenerate(Selection.at["slope"] | Selection.at["intercept"])


def make_kernel(model):
    chm = ChoiceMap.d({"ys": YS})

    def step(i, carry):
        tr, key, acc = carry
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, REQ)
        w = w - (new_tr.project(k1, REQ.selection) - tr.project(k1, REQ.selection))
        a = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a, lambda x, y: x, lambda x, y: y, new_tr, tr)
        return (tr, key, acc + a)

    @jax.jit
    def kernel(key):
        kg, kl = jax.random.split(key)
        tr, _ = model.importance(kg, chm, (XS, SIG))
        tr, _, acc = jax.lax.fori_loop(0, NUM_SWEEPS, step, (tr, kl, jnp.array(0)))
        ch = tr.get_choices()
        return ch["slope"], ch["intercept"], acc

    return kernel


def lp(x, mu, s):
    return -0.5 * np.log(2 * np.pi) - np.log(s) - (x - mu) ** 2 / (2 * s * s)


slope0, intercept0 = -0.9, 1.8
xs_np, ys_np = np.array(d["xs"]), np.array(d["ys"])
hand = lp(slope0, 0.0, 1.0) + lp(intercept0, 0.0, 2.0) + np.sum(lp(ys_np, slope0 * xs_np + intercept0, 0.1))

print(f"config: row {ROW} N={N} sweeps={NUM_SWEEPS} reps={NREPS} encodings={','.join(ENCODINGS)} "
      f"jax={jax.__version__} devices={jax.devices()}", flush=True)
for enc in ENCODINGS:
    model = MODELS[enc]
    kernel = make_kernel(model)
    label = f"GENJAX linreg_tutorial_{ROW} {enc}"
    key = jax.random.PRNGKey(1)
    out = kernel(key); jax.block_until_ready(out)          # warm-up, discarded
    walls = []
    for _ in range(NREPS):
        t0 = time.perf_counter()
        out = kernel(key); jax.block_until_ready(out)
        t = time.perf_counter() - t0
        walls.append(t)
        print(f"{label} REP wall {t:.6f} s | per-move {t / NUM_SWEEPS * 1e6:.4f} us", flush=True)
    s, i, acc = (float(out[0]), float(out[1]), int(out[2]))
    print(f"{label} per-move median {statistics.median(walls) / NUM_SWEEPS * 1e6:.4f} us "
          f"min {min(walls) / NUM_SWEEPS * 1e6:.4f} us | accepts {acc}/{NUM_SWEEPS} | "
          f"slope={s:.10f} intercept={i:.10f}", flush=True)
    chm_fixed = ChoiceMap.d({"ys": YS, "slope": jnp.float64(slope0), "intercept": jnp.float64(intercept0)})
    score, _ = model.assess(chm_fixed, (XS, SIG))
    print(f"{label} joint density: genjax={float(score):.12f} hand={hand:.12f} |diff|={abs(float(score) - hand):.3e}", flush=True)

# Prior invariance of the block move (no data): the same step on a prior-only
# model (slope, intercept, no ys site). A correct prior-resimulation MH accepts
# every proposal and leaves the prior invariant: slope mean 0 var 1, intercept
# mean 0 var 4. The raw Regenerate weight (prior counted twice) targets prior^2:
# variances 0.5 and 2. Both acceptances are run so the log shows the difference.
PI_SWEEPS = int(os.environ.get("GENJAX_LT_PI_SWEEPS", "200000"))


@genjax.gen
def line_prior():
    slope = genjax.normal(jnp.float64(0.0), jnp.float64(1.0)) @ "slope"
    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "intercept"
    return slope


def prior_check(corrected):
    def step(i, carry):
        tr, key, m1, m2, nacc = carry
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, REQ)
        if corrected:
            w = w - (new_tr.project(k1, REQ.selection) - tr.project(k1, REQ.selection))
        a = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a, lambda x, y: x, lambda x, y: y, new_tr, tr)
        ch = tr.get_choices()
        v = jnp.array([ch["slope"], ch["intercept"]])
        return (tr, key, m1 + v, m2 + v * v, nacc + a)

    @jax.jit
    def run(key):
        kg, kl = jax.random.split(key)
        tr = line_prior.simulate(kg, ())
        _, _, m1, m2, nacc = jax.lax.fori_loop(0, PI_SWEEPS, step, (tr, kl, jnp.zeros(2), jnp.zeros(2), jnp.array(0)))
        return m1, m2, nacc

    m1, m2, nacc = jax.device_get(run(jax.random.PRNGKey(2)))
    mu = m1 / PI_SWEEPS
    va = m2 / PI_SWEEPS - mu ** 2
    tag = "corrected acceptance (the timed kernel)" if corrected else "raw Regenerate weight (NOT used)"
    print(f"GENJAX linreg_tutorial_{ROW} prior invariance [{tag}]: sweeps={PI_SWEEPS} accepts={int(nacc)} "
          f"slope mean={mu[0]:.4f} var={va[0]:.4f} intercept mean={mu[1]:.4f} var={va[1]:.4f} "
          f"(correct 0/1, 0/4; prior^2 target 0/0.5, 0/2)", flush=True)


if PI_SWEEPS > 0:
    prior_check(True)
    prior_check(False)
