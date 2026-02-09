"""
NumPyro NUTS baseline for Comrade posterior.

Uses the same serialized logdensity/gradient from enzyme_ad as main.py (BlackJAX),
but runs NumPyro's NUTS in potential_fn mode to mirror Reactant's implementation.

Hyperparameters are matched to comimager_mcmc_logpdf.jl:
  - step_size = 1e-3
  - max_tree_depth = 10
  - dense_mass = False (diagonal)
  - inverse_mass_matrix = ones (diagonal)
  - adapt_step_size = True
  - adapt_mass_matrix = True
  - find_heuristic_step_size = False
  - 200 warmup + 100 samples
"""
import sys
import time
import os

sys.path.insert(0, "Serialized/Fwd")
sys.path.insert(0, "Serialized/Bwd")

os.environ["NUMPYRO_DISABLE_CONTROL_FLOW_PRIM"] = "1"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import enzyme_ad

import numpyro
import numpyro.infer as infer
from numpyro.diagnostics import effective_sample_size

import logdensityof as lg
import gl as gl

from logdensityof import run_logdensityof
from gl import run_gl

NUM_WARMUP = 200
NUM_SAMPLES = 100

print(f"Devices: {jax.devices()}")

lg_inputs = lg.load_inputs()
gl_inputs = gl.load_inputs()

tpost = lg_inputs[:-1]
xr = lg_inputs[-1]

jtpost = tuple(jnp.array(t) for t in tpost)
jxr = jnp.array(xr)
pos_size = len(jxr)

print(f"Position size: {pos_size}")

# Warm up JIT caches
run_logdensityof(*jtpost, jxr)
run_gl(*jtpost, jxr)


# Build logdensity with custom VJP (same as main.py)
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

# NumPyro uses potential_fn = -logdensity (potential energy = negative log density)
def potential_fn(z):
    return -f(z["x"])


# Match Reactant's configuration exactly:
#   seed = [42, 0], step_size = 1e-3, diagonal mass matrix
init_params = {"x": jxr}
inverse_mass_matrix = jnp.ones(pos_size)

kernel = infer.NUTS(
    potential_fn=potential_fn,
    step_size=1e-3,
    max_tree_depth=10,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=False,
    inverse_mass_matrix=inverse_mass_matrix,
    find_heuristic_step_size=False,
)

mcmc = infer.MCMC(
    kernel,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    progress_bar=False,
)

# Use PRNGKey(42) to be close to Reactant's seed [42, 0]
rng_key = jax.random.PRNGKey(42)

print(f"\nRunning NumPyro NUTS ({NUM_WARMUP} warmup + {NUM_SAMPLES} samples)...")
t0 = time.time()
mcmc.run(rng_key, init_params=init_params)
run_time = time.time() - t0
print(f"Total time: {run_time:.1f} s")

mcmc.print_summary(exclude_deterministic=False)

# Extract samples for custom summary
samples = np.array(mcmc.get_samples()["x"])
print(f"\nSamples shape: {samples.shape}")

# Adapted parameters
last_state = mcmc.last_state
if hasattr(last_state, "adapt_state") and last_state.adapt_state is not None:
    adapt = last_state.adapt_state
    if hasattr(adapt, "step_size"):
        print(f"Adapted step size: {float(adapt.step_size):.6f}")

# Summary statistics (matching main.py format)
samples_3d = samples[np.newaxis, ...]
ess = np.array(effective_sample_size(samples_3d))

print(f"\n{'':>10s} {'mean':>10s} {'std':>10s} {'median':>10s} {'5.0%':>10s} {'95.0%':>10s} {'n_eff':>10s}")
print("-" * 75)
n_params = samples.shape[1]
show_indices = list(range(min(20, n_params)))
if n_params > 25:
    show_indices += list(range(n_params - 5, n_params))
for i in show_indices:
    if i == 20 and n_params > 25:
        print(f"  ... ({n_params - 25} parameters omitted) ...")
    col = samples[:, i]
    print(
        f"{'x[' + str(i) + ']':>10s} {np.mean(col):10.2f} {np.std(col, ddof=1):10.2f} "
        f"{np.median(col):10.2f} {np.percentile(col, 5):10.2f} {np.percentile(col, 95):10.2f} "
        f"{ess[i]:10.1f}"
    )
