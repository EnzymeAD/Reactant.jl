import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import genjax
from genjax import ChoiceMap, Regenerate, Selection

N = 20
X = jnp.array([-3.0, -2.6842, -2.3684, -2.0526, -1.7368, -1.4211, -1.1053, -0.7895, -0.4737, -0.1579, 0.1579, 0.4737, 0.7895, 1.1053, 1.4211, 1.7368, 2.0526, 2.3684, 2.6842, 3.0])
Y = jnp.array([16.5608, 15.9844, 14.4534, 13.875, 35.0582, 12.9817, 12.0972, 11.4781, 11.3252, 10.3109, 9.7693, 9.9745, -10.6448, 6.9002, 7.2688, 6.3987, 5.8039, 5.0438, 22.2934, 3.4963])

def make_model():
    body = ["@genjax.gen", "def outlier_model(xs):",
            '    slope = genjax.normal(jnp.float64(0.0), jnp.float64(2.0)) @ "slope"',
            '    intercept = genjax.normal(jnp.float64(0.0), jnp.float64(10.0)) @ "intercept"']
    for i in range(N):
        body.append(f'    o{i} = genjax.flip(jnp.float64(0.1)) @ "o{i}"')
    flags = ", ".join(f"o{i}" for i in range(N))
    body += [f'    ov = jnp.stack([{flags}]).astype(jnp.float64)',
             '    sig = 1.0 + 9.0 * ov',
             '    ys = genjax.mv_normal_diag(slope * xs + intercept, sig) @ "ys"',
             '    return ys']
    ns = {"genjax": genjax, "jnp": jnp}
    exec("\n".join(body), ns)
    return ns["outlier_model"]

outlier_model = make_model()
chm = ChoiceMap.d({"ys": Y, "slope": jnp.float64(-2.0), "intercept": jnp.float64(10.0)})
reqs = [Regenerate(Selection.at["slope"]), Regenerate(Selection.at["intercept"])] + \
       [Regenerate(Selection.at[f"o{i}"]) for i in range(N)]
NUM_SWEEPS = 200
MOVES = NUM_SWEEPS * (N + 2)

def step(i, carry):
    tr, key = carry
    for req in reqs:
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = tr.edit(k1, req)
        a = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.lax.cond(a, lambda x, y: x, lambda x, y: y, new_tr, tr)
    return (tr, key)

@jax.jit
def kernel(key):
    kg, kl = jax.random.split(key)
    tr, _ = outlier_model.importance(kg, chm, (X,))
    tr, _ = jax.lax.fori_loop(0, NUM_SWEEPS, step, (tr, kl))
    ch = tr.get_choices()
    flags = jnp.stack([ch[f"o{i}"] for i in range(N)])
    return ch["slope"], flags

key = jax.random.PRNGKey(777)
out = kernel(key); jax.block_until_ready(out)
best = float("inf")
for _ in range(3):
    t0 = time.perf_counter()
    out = kernel(key); jax.block_until_ready(out)
    best = min(best, time.perf_counter() - t0)
slope, flags = jax.device_get(out[0]), np.array(jax.device_get(out[1]))
idx = ",".join(str(i + 1) for i in np.where(flags > 0.5)[0])
print(f"GENJAX outlier per-move {best/MOVES*1e6:.3f} us | slope = {float(slope):.3f} | flags = {idx}")
