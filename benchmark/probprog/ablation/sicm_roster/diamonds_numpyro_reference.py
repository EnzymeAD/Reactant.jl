#!/usr/bin/env python3
"""NumPyro reference model for the diamonds regression row, and its joint log
density at given parameter points (the NumPyro agreement check of the row).

posteriordb `diamonds-diamonds` (brms 2.10.0), written exactly as
sicm/diamonds_regression.jl writes it (intercept folded into the augmented
design, X = [1, Xc]):

    intercept ~ StudentT(3, 8, 10)
    b         ~ Normal(0, 1)^(K-1)
    sigma     ~ TruncatedDistribution(StudentT(3, 0, 10), low=0)
                (Stan: student_t_lpdf(sigma | 3, 0, 10)
                       - student_t_lccdf(0 | 3, 0, 10))
    y         ~ Normal(X @ [intercept; b], sigma)

Usage:
  diamonds_numpyro_reference.py <data.json> <points.json> <out.json>

points.json: a list of {"beta": [K floats], "sigma": [1 float]} (constrained
space). out.json: {"log_density": [...], "sigma_prior": [...]} per point, the
joint log density (numpyro.infer.util.log_density, jitted, no Jacobian) and
the sigma prior term alone.
"""
import json
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402


def sigma_prior():
    return dist.TruncatedDistribution(dist.StudentT(3.0, 0.0, 10.0), low=0.0)


def model(X, Y):
    K = X.shape[1]
    intercept = numpyro.sample("intercept", dist.StudentT(3.0, 8.0, 10.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([K - 1]).to_event(1))
    sigma = numpyro.sample("sigma", sigma_prior())
    beta = jnp.concatenate([jnp.reshape(intercept, (1,)), b])
    numpyro.sample("y", dist.Normal(X @ beta, sigma).to_event(1), obs=Y)


def main():
    data_path, points_path, out_path = sys.argv[1:4]
    with open(data_path) as f:
        data = json.load(f)
    X = jnp.asarray(data["X"], dtype=jnp.float64)
    Y = jnp.asarray(data["Y"], dtype=jnp.float64)
    with open(points_path) as f:
        points = json.load(f)

    @jax.jit
    def joint(intercept, b, sigma):
        params = {"intercept": intercept, "b": b, "sigma": sigma}
        return log_density(model, (X, Y), {}, params)[0]

    @jax.jit
    def sprior(sigma):
        return sigma_prior().log_prob(sigma)

    ld, sp = [], []
    for p in points:
        beta = jnp.asarray(p["beta"], dtype=jnp.float64)
        sigma = jnp.asarray(p["sigma"][0], dtype=jnp.float64)
        ld.append(float(joint(beta[0], beta[1:], sigma)))
        sp.append(float(sprior(sigma)))
    with open(out_path, "w") as f:
        json.dump({"log_density": ld, "sigma_prior": sp,
                   "jax": jax.__version__, "numpyro": numpyro.__version__}, f)


if __name__ == "__main__":
    main()
