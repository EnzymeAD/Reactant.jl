"""NumPyro SVI reference for a NO-OBS model:  x ~ Normal(0, 1).

Analytically the ELBO per iteration is:
    elbo(eps) = log_p(x) - log_q(x;loc,scale)
             = (-0.5*log(2pi) - 0.5*x^2) - (-log(scale) - 0.5*eps^2 - 0.5*log(2pi))
             = log(scale) + 0.5*eps^2 - 0.5*x^2

At init loc=0, scale=0.1:  elbo ≈ log(0.1) + 0.5*eps^2 - 0.005*eps^2 ≈ -2.303 + 0.495*eps^2
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam


def model():
    numpyro.sample("x", dist.Normal(0.0, 1.0))


def main():
    guide = AutoNormal(
        model, init_loc_fn=init_to_value(values={"x": 0.0}), init_scale=0.1,
    )
    optimizer = Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))
    state = svi.init(random.PRNGKey(0))

    print("step\tloc_u\tscale_u\tscale\telbo")
    print(f"0\t0\t{float(jnp.log(jnp.expm1(0.1))):.17g}\t0.1\tnan")

    for step in range(1, 4):
        state, loss = svi.update(state)
        elbo = -float(loss)
        params = svi.get_params(state)
        _, (states_flat, _, _) = state.optim_state
        (loc_x, _, _), (scale_u, _, _) = states_flat
        scale_c = float(jax.nn.softplus(scale_u))
        print(f"{step}\t{float(loc_x):.17g}\t{float(scale_u):.17g}\t{scale_c:.17g}\t{elbo:.17g}")


if __name__ == "__main__":
    main()
