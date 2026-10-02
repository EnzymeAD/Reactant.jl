"""
Acceptance test for the cochlear-aqueduct diffusion row (validity criterion of
the pre-lowering ablation): gradient-only cost of the NAIVE program (operator
construction + Cholesky inside the time loop, as the published CUQIpy
implementation writes it) against the HAND-HOISTED program (construction and
factorization once per phase, triangular solves in the loop). Both are given to
XLA through jax.jit(jax.grad(.)). If XLA closes the gap, the row fails.

Same model as cochlear_diffusion.jl (joint log density over d in R^21 and
z = log delta), same exported data (cochlear_data/cochlear_<animal>_<ear>.json).

Usage: python cochlear_jax_gradient.py [data.json] [n_reps]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular

jax.config.update("jax_enable_x64", True)

path = sys.argv[1] if len(sys.argv) > 1 else "cochlear_data/cochlear_m1_r.json"
N_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
D = json.load(open(path))
n, nt = D["n"], D["nt"]
h = D["h"]
Ic = jnp.array(D["Ic"]); Dx = jnp.array(D["Dx"])
bl = jnp.array(D["bl"]); br = jnp.array(D["br"]); u0 = jnp.array(D["u0"])
Px = jnp.array(D["Px"]); PtT = jnp.array(D["Pt"]).T  # nt x 7
Gfd = jnp.array(D["Gfd"]); Y = jnp.array(D["data"])
dts = D["dts"]
n_small = sum(1 for d in dts if d < 0.5 * max(dts))
DT_S = dts[0]; DT_L = dts[-1]
n_large = nt - 1 - n_small
GP = Gfd @ Px  # 4 x n
I_n = jnp.eye(n)
e0 = jnp.zeros(n).at[0].set(1.0); eN = jnp.zeros(n).at[-1].set(1.0)
nobs = Y.size
mu, sd, ga, gb = D["prior_d_mean"], D["prior_d_std"], D["gamma_shape"], D["gamma_rate"]


def chol_solve_lower(Lc, b):
    y = solve_triangular(Lc, b, lower=True)
    return solve_triangular(Lc.T, y, lower=False)


def trajectory_obs(cf, naive):
    # acc (4 x 7) = sum_k (G Px u_k) Pt[:, k]'
    acc = jnp.outer(GP @ u0, PtT[0])

    def phase(u, acc, k0, nsteps, dt):
        if not naive:
            A = I_n + dt * (Dx.T @ (cf[:, None] * Dx))
            Lc = jnp.linalg.cholesky(A)

        def body(i, carry):
            u, acc = carry
            k = k0 + i
            if naive:
                A = I_n + dt * (Dx.T @ (cf[:, None] * Dx))
                Lc_ = jnp.linalg.cholesky(A)
            else:
                Lc_ = Lc
            rhs = u + dt * (cf[0] / h**2 * bl[k] * e0 + cf[-1] / h**2 * br[k] * eN)
            u = chol_solve_lower(Lc_, rhs)
            acc = acc + jnp.outer(GP @ u, lax.dynamic_index_in_dim(PtT, k + 1, keepdims=False))
            return (u, acc)

        return lax.fori_loop(0, nsteps, body, (u, acc))

    u, acc = phase(u0, acc, 0, n_small, DT_S)
    u, acc = phase(u, acc, n_small, n_large, DT_L)
    return acc


def logjoint(theta, naive):
    d = theta[:-1]; z = theta[-1]
    delta = jnp.exp(z)
    cf = Ic @ (d * d)
    pred = trajectory_obs(cf, naive)
    r = Y - pred
    loglik = 0.5 * nobs * (z - jnp.log(2 * jnp.pi)) - 0.5 * delta * jnp.sum(r * r)
    logprior_d = -0.5 * jnp.sum(((d - mu) / sd) ** 2) - d.size * (jnp.log(sd) + 0.5 * jnp.log(2 * jnp.pi))
    # delta ~ Gamma(shape ga, rate gb) with the log-Jacobian of delta = exp(z)
    logprior_z = ga * jnp.log(gb) - jax.scipy.special.gammaln(ga) + (ga - 1) * z - gb * delta + z
    return loglik + logprior_d + logprior_z


g_naive = jax.jit(jax.grad(lambda t: logjoint(t, True)))
g_hoist = jax.jit(jax.grad(lambda t: logjoint(t, False)))
theta = jnp.concatenate([jnp.full(21, 20.0) + 0.5 * jnp.sin(jnp.arange(21.0)), jnp.array([jnp.log(0.1)])])

a = g_naive(theta).block_until_ready(); b = g_hoist(theta).block_until_ready()
print(f"data: {path}  n={n} steps={nt-1} ({n_small} x {DT_S:.4f} s + {n_large} x {DT_L:.4f} s)")
print(f"grad max rel diff naive vs hoisted: {float(jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(b))):.3e}")

# Source-parity check of the log density (joint over d and delta, no Jacobian term).
for rp in D["reference_points"]:
    th = jnp.concatenate([jnp.array(rp["d"]), jnp.array([jnp.log(rp["s"])])])
    lj = float(jax.jit(lambda t: logjoint(t, True))(th)) - float(th[-1])  # drop log-Jacobian
    print(f"  logd impl {lj:.12f}  source {rp['joint_logd']:.12f}  diff {lj-rp['joint_logd']:.3e}")


def bench(f):
    f(theta).block_until_ready()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            f(theta).block_until_ready()
        ts.append((time.perf_counter() - t0) / N_REPS)
    ts.sort()
    return ts[2], ts


tn, tsn = bench(g_naive)
th_, tsh = bench(g_hoist)
print(f"naive gradient:   median {tn*1e6:.1f} us  all {[round(t*1e6,1) for t in tsn]}")
print(f"hoisted gradient: median {th_*1e6:.1f} us  all {[round(t*1e6,1) for t in tsh]}")
print(f"gap (naive/hoisted): {tn/th_:.2f}x")

for name, f in (("naive", g_naive), ("hoisted", g_hoist)):
    txt = f.lower(theta).compile().as_text()
    print(f"[{name}] optimized HLO: lines={txt.count(chr(10))} while={txt.count(' while(')} "
          f"cholesky/potrf={txt.count('cholesky') + txt.count('potrf')} "
          f"triangular-solve={txt.count('triangular-solve') + txt.count('trsm')}")
    open(f"/tmp/sbrantq-fig9/jax_{name}_hlo.txt", "w").write(txt)
