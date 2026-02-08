import sys

sys.path.insert(0, "Serialized/Fwd")
sys.path.insert(0, "Serialized/Bwd")


import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import blackjax
import enzyme_ad
from functools import partial

import logdensityof as lg
import gl as gl

from logdensityof import run_logdensityof
from gl import run_gl

lg_inputs = lg.load_inputs()
gl_inputs = gl.load_inputs()

tpost = lg_inputs[:-1]
xr = lg_inputs[-1]

jlr = jax.jit(run_logdensityof)

jtpost = tuple(jnp.array(t) for t in tpost)
jxr = jnp.array(xr)

out = jlr(*jtpost, jxr)

run_logdensityof(*jtpost, jxr)
run_gl(*jtpost, jxr)


@jax.custom_vjp
def f(x):
    out = run_logdensityof(*jtpost, x)
    return out[0]


def f_fwd(x):
    j = run_gl(*jtpost, x)[0]
    return f(x), (j,)


def f_bwd(res, g):
    j = res[0]
    return (g * j,)


f.defvjp(f_fwd, f_bwd)

logdensity = lambda x: f(**x)

inv_mass_matrix = jnp.ones(len(jxr))
initial_position = {"x": jxr}

rng_key, sample_key = jax.random.split(jax.random.PRNGKey(0))

# adaptation
warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, progress_bar=True)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
(state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)


def inference_loop(rng_key, kernel, init, nsamples):
    @jax.jit
    def step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, nsamples)
    _, states = jax.lax.scan(step, init, keys)
    return states


# inference loop
rng_key, sample_key = jax.random.split(jax.random.PRNGKey(0))
kernel = blackjax.nuts(logdensity, **parameters).step
states = inference_loop(sample_key, kernel, state, nsamples=1000)
