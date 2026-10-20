"""
Acceptance test for the IDE radar row (pre-lowering ablation, Fig. 9 panel a): gradient-only
cost of the NAIVE program (propagator M(theta) built inside the Kalman-filter time loop) against
the HAND-HOISTED program (M built once before the loop). Both go through jax.jit(jax.grad(.)),
so XLA gets every chance to hoist after AD. If XLA closes the gap, the row fails.

Model: integro-difference equation (IDE) of Wikle, Zammit-Mangion and Cressie (2019),
"Spatio-Temporal Statistics with R", Lab 5.2 (R package IDE, spatially invariant kernel) on the
Sydney radar data (STRbook::radar_STIDF), exported from STRbook::IDE_Radar_results by
export_ide_radar.R. The IDE package evaluates the marginal likelihood through a joint sparse
precision over all time points; here the same marginal likelihood is computed by a Kalman filter
over the 14 modelled time points (2 hindcast steps without data, 12 data steps), in information
form (the 1120 pixels per image enter through PHI'PHI and PHI'z_t).

    k(s, x) = k1 exp(-|x - s - (k3, k4)|^2 / k2)            (kernel on the 41 x 41 grid)
    M(theta) = GRAM^-1 PHI_g' K PHI_g area^2                 (r x r propagator, r = 129)
    alpha_0 ~ N(0, s2eta I),  alpha_t = M alpha_{t-1} + eta_t,  eta_t ~ N(0, s2eta I)
    z_t = beta + PHI alpha_t + eps_t,  eps_t ~ N(0, s2eps I)   (t = 2..13)
Position: (log k1, log k2, k3, k4, log s2eps, log s2eta, beta), normal priors whose central 95%
interval is the IDE package's DEoptim search box (plausible_ranges); beta ~ N(0, 10^2).

Usage: python ide_radar_jax_gradient.py <ide_radar.json> [grid_stride] [n_reps]
  grid_stride 1 = the package default grid_size 41; 2 = every other point = IDE(grid_size = 21).
"""
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

path = sys.argv[1]
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
D = json.load(open(path))

r = D["r"]
sg_full = np.array(D["s_grid"])            # n_grid x 2, columns (s2, s1) as in the package
PHIg_full = np.array(D["PHI_grid"])        # n_grid x r
G = D["grid_size"]
if STRIDE == 1:
    sg, PHIg, area = sg_full, PHIg_full, D["area"]
else:
    idx = np.arange(G * G).reshape(G, G)[::STRIDE, ::STRIDE].ravel()
    sg, PHIg = sg_full[idx], PHIg_full[idx]
    gs = (G - 1) // STRIDE + 1
    area = D["area"] * (G / gs) ** 2      # IDE package construct_grid(bbox, gs): area = prod(range / gs)
ng = sg.shape[0]
GRAM = PHIg.T @ PHIg * area
GRAMinv = np.linalg.inv(GRAM)
PHI1 = np.array(D["PHI_obs_block"])        # 1120 x r
Zd = np.array(D["Z"])                      # 1120 x 12 (column = data time)
n = PHI1.shape[0]
dti = np.array(D["data_time_index"])       # 2..13
NT = int(dti.max()) + 1                    # 14 modelled steps carry likelihood
Zall = np.zeros((NT, n)); W = np.zeros(NT)
Zall[dti] = Zd.T; W[dti] = 1.0
A = PHI1.T @ PHI1                          # data-only information matrix
PZ = Zall @ PHI1                           # NT x r, PHI' z_t
S1 = PHI1.sum(axis=0)                      # PHI' 1
ZZ = (Zall * Zall).sum(axis=1); SZ = Zall.sum(axis=1)

PR = D["plausible_ranges"]
def box(lo, hi):
    return (0.5 * (lo + hi), 0.25 * (hi - lo))
PRI = [box(np.log(PR[0]["k1"]), np.log(PR[1]["k1"])), box(np.log(PR[0]["k2"]), np.log(PR[1]["k2"])),
       box(PR[0]["k3"], PR[1]["k3"]), box(PR[0]["k4"], PR[1]["k4"]),
       box(np.log(PR[0]["sigma2_eps"]), np.log(PR[1]["sigma2_eps"])),
       box(np.log(PR[0]["sigma2_eta"]), np.log(PR[1]["sigma2_eta"])), (0.0, 10.0)]
PMU = jnp.array([p[0] for p in PRI]); PSD = jnp.array([p[1] for p in PRI])

sg_j = jnp.array(sg); PHIg_j = jnp.array(PHIg); GRAMinv_j = jnp.array(GRAMinv)
A_j = jnp.array(A); PZ_j = jnp.array(PZ); S1_j = jnp.array(S1)
ZZ_j = jnp.array(ZZ); SZ_j = jnp.array(SZ); W_j = jnp.array(W)
I_r = jnp.eye(r)
LOG2PI = np.log(2 * np.pi)


def build_M(th):
    k1, k2, k3, k4 = jnp.exp(th[0]), jnp.exp(th[1]), th[2], th[3]
    d0 = (sg_j[:, 0:1] + k3) - sg_j[:, 0][None, :]
    d1 = (sg_j[:, 1:2] + k4) - sg_j[:, 1][None, :]
    K = k1 * jnp.exp(-(d0 * d0 + d1 * d1) / k2)          # ng x ng, rows s, columns x
    return GRAMinv_j @ (PHIg_j.T @ (K @ PHIg_j)) * (area * area)


def kf_step(M, s2eps, s2eta, beta, carry, xs):
    m, P, ll = carry
    pz, zz, sz, w = xs
    m = M @ m                                            # predict
    P = M @ P @ M.T + s2eta * I_r
    LP = jnp.linalg.cholesky(P)
    Aw = w * A_j
    Gm = I_r + (LP.T @ Aw @ LP) / s2eps
    LG = jnp.linalg.cholesky(Gm)
    g = w * (pz - beta * S1_j) - Aw @ m                  # PHI'(z - beta - PHI m)
    ee = w * (zz - 2 * beta * sz + beta * beta * n) - 2 * m @ (w * (pz - beta * S1_j)) + m @ Aw @ m
    u = LP.T @ g
    y = solve_triangular(LG, u, lower=True)
    v = solve_triangular(LG.T, y, lower=False)
    quad = (ee - u @ v / s2eps) / s2eps
    logdetG = 2 * jnp.sum(jnp.log(jnp.diag(LG)))
    ll = ll - 0.5 * (w * n * (LOG2PI + jnp.log(s2eps)) + logdetG + quad)
    m = m + LP @ v / s2eps                               # update
    Wm = solve_triangular(LG, LP.T, lower=True)
    P = Wm.T @ Wm
    return (m, P, ll), None


def loglik(th, naive):
    s2eps, s2eta, beta = jnp.exp(th[4]), jnp.exp(th[5]), th[6]
    init = (jnp.zeros(r), jnp.zeros((r, r)), jnp.zeros(()))
    xs = (PZ_j, ZZ_j, SZ_j, W_j)
    if naive:
        def body(c, x):
            M = build_M(th)                              # loop-invariant, parameter-dependent
            return kf_step(M, s2eps, s2eta, beta, c, x)
    else:
        M = build_M(th)
        def body(c, x):
            return kf_step(M, s2eps, s2eta, beta, c, x)
    (_, _, ll), _ = lax.scan(body, init, xs)
    return ll


def logjoint(th, naive):
    zp = (th - PMU) / PSD
    return loglik(th, naive) - 0.5 * jnp.sum(zp * zp) - jnp.sum(jnp.log(PSD)) - 3.5 * LOG2PI


def theta_of(p):
    k = p["k"]
    return jnp.array([np.log(k[0]), np.log(k[1]), k[2], k[3], np.log(p["s2eps"]), np.log(p["s2eta"]), p["beta"]])


print(f"jax {jax.__version__}  devices {jax.devices()}  grid stride {STRIDE}: n_grid={ng}  r={r}  steps={NT}  n/time={n}")
if STRIDE == 1:
    M_fit = np.array(build_M(theta_of(D["reference_points"][0])))
    print(f"max |M(theta_fit) - M_IDE_package| = {np.abs(M_fit - np.array(D['M_fit'])).max():.3e}")
    for rp in D["reference_points"]:
        th = theta_of(rp)
        a = float(jax.jit(lambda t: loglik(t, True))(th)); b = float(jax.jit(lambda t: loglik(t, False))(th))
        print(f"  loglik KF naive {a:.10f} hoisted {b:.10f}  IDE-package joint precision (R) {rp['loglik']:.10f}  diff {a - rp['loglik']:.3e}")

g_naive = jax.jit(jax.grad(lambda t: logjoint(t, True)))
g_hoist = jax.jit(jax.grad(lambda t: logjoint(t, False)))
theta = theta_of(D["reference_points"][0]) + 0.01 * jnp.arange(7.0)
a = g_naive(theta).block_until_ready(); b = g_hoist(theta).block_until_ready()
print(f"grad max rel diff naive vs hoisted: {float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))):.3e}")


def bench(f, reps):
    f(theta).block_until_ready()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            f(theta).block_until_ready()
        ts.append((time.perf_counter() - t0) / reps)
    ts.sort()
    return ts[2], ts


f_hoist = jax.jit(lambda t: logjoint(t, False))
f_M = jax.jit(build_M)
g_M = jax.jit(jax.grad(lambda t: jnp.sum(build_M(t) * jnp.arange(r * r).reshape(r, r))))
tn, tsn = bench(g_naive, max(1, N_REPS // 4))
th_, tsh = bench(g_hoist, N_REPS)
tf, _ = bench(f_hoist, N_REPS)
tM, _ = bench(f_M, N_REPS)
tgM, _ = bench(g_M, N_REPS)
print(f"naive gradient:   median {tn*1e3:.2f} ms  all {[round(t*1e3,2) for t in tsn]}")
print(f"hoisted gradient: median {th_*1e3:.2f} ms  all {[round(t*1e3,2) for t in tsh]}")
print(f"gap (naive/hoisted): {tn/th_:.2f}x")
print(f"components: hoisted forward {tf*1e3:.2f} ms; build_M forward {tM*1e3:.2f} ms; build_M fwd+rev (VJP) {tgM*1e3:.2f} ms")
pred = (tgM + NT * (tgM - tM) + (th_ - tgM)) / th_
print(f"component prediction (hoisted + (steps-1) x build_M reverse): {pred:.2f}x")

for name, f in (("naive", g_naive), ("hoisted", g_hoist)):
    txt = f.lower(theta).compile().as_text()
    print(f"[{name}] optimized HLO: lines={txt.count(chr(10))} while={txt.count(' while(')} "
          f"exp={txt.count(' exponential(')} dot={txt.count(' dot(')} "
          f"cholesky/potrf={txt.count('cholesky(') + txt.count('potrf')}")
