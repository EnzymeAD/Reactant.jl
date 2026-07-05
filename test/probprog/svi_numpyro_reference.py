#!/usr/bin/env python
# NumPyro SVI (AutoNormal + Adam, svi.run) throughput reference.
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

numpyro.set_host_device_count(1)
N = 8
NUM_STEPS = 300
LR = 0.01


def data_mvn(n):
    R = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                R[i, j] = 0.3 ** abs(i - j)
    y = np.linspace(0.4, -0.4, n)
    return jnp.array(R), jnp.array(y)


def data_gp(n):
    x = jnp.array(np.linspace(1.0, float(n), n))
    rho = 1.0
    k = jnp.array(np.array([(i % 3) + 1 for i in range(1, n + 1)], dtype=float))
    return x, rho, k


def model_mvn(R, y):
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(N),
                   covariance_matrix=tau ** 2 * R), obs=y)


def model_gp(x, rho, k):
    alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
    f_tilde = numpyro.sample("f_tilde", dist.Normal(0, 1).expand([N]))
    delta = x[:, None] - x[None, :]
    Kb = jnp.exp(-0.5 * (delta / rho) ** 2)
    L = jnp.linalg.cholesky(alpha ** 2 * Kb)
    f = L @ f_tilde
    numpyro.sample("k", dist.Poisson(rate=jnp.exp(f)), obs=k)


def run(name, model, args, init_vals):
    guide = AutoNormal(model, init_loc_fn=init_to_value(values=init_vals),
                       init_scale=0.1)
    opt = numpyro.optim.Adam(step_size=LR)
    svi = SVI(model, guide, opt, loss=Trace_ELBO(num_particles=1))
    key = jax.random.PRNGKey(0)

    def once():
        res = svi.run(key, NUM_STEPS, *args, progress_bar=False)
        res.losses.block_until_ready()
        return res

    res = once()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        r = once()
        r.losses.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    losses = np.asarray(res.losses)
    final_elbo = float(-losses[-1])
    per_step_us = best / NUM_STEPS * 1e6
    print(f"[{name}] numpyro svi.run: final_elbo={final_elbo:.6f} "
          f"wall={best*1e3:.2f}ms per_step={per_step_us:.2f}us")
    return {"final_elbo": final_elbo, "per_step_us": per_step_us,
            "wall_ms": best * 1e3, "elbo_traj": (-losses).tolist()}


out = {}
R, y = data_mvn(N)
out["scale_family_mvn"] = run("scale_family_mvn", model_mvn, (R, y),
                              {"tau": 1.0})
x, rho, k = data_gp(N)
out["gp_pois_regr"] = run("gp_pois_regr", model_gp, (x, rho, k),
                          {"alpha": 1.0, "f_tilde": np.zeros(N)})

with open("/tmp/claude-140061798/-mnt-sbrantq-probprog/9007adff-5119-4ba4-8ced-2cd50c62b05e/scratchpad/sdd/d3-evidence/numpyro_svi_fair.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote numpyro_svi_fair.json")
