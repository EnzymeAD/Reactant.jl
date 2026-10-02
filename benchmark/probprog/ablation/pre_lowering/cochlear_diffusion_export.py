#!/usr/bin/env python
"""Export the published cochlear-aqueduct diffusion model (Alghamdi et al., JARO
2026, doi:10.1007/s10162-026-01047-x) for the Impulse pre-lowering harness.

The problem is built by the SOURCE code itself (CUQI-DTU/Paper-Cochlear v1.1.0,
Zenodo doi:10.5281/zenodo.15796673, CUQIpy 1.5.0), following
code/advection_diffusion_inference.py steps 1-16 verbatim for the paper's
real-data diffusion-only configuration (real_data_results.ipynb):

  -data_type real -inference_type heterogeneous -num_CA 5 -num_ST 0
  -rbc fromDataClip --data_grad --adaptive --u0_from_data  (NUTSWithGibbs)

Every fixed ingredient of the forward map is then extracted as a dense linear
map by probing the source objects with unit vectors, so nothing is
re-derived by hand:

  Ic   (n+1) x 21   np.interp coarse -> fine diffusivity map (PDE_form)
  Dx   (n+1) x n    FD derivative matrix (create_PDE_form)
  bl, br            per-step Dirichlet boundary values np.interp(t, times, bc)
  Px   5 x n, Pt 7 x 19   separable factors of the RectBivariateSpline observation
                    map (verified rank-1 per probe to 1e-12 relative)
  G    4 x 5        FD_spatial_gradient
  u0, data, prior, Gamma hyper-parameters.

It also records reference values from the source for parity checks: the model
output A(d) and the CUQIpy joint log density at fixed points.

Usage: python cochlear_diffusion_export.py <Paper-Cochlear dir> <out.json> [animal] [ear]
"""
import json
import os
import sys
from copy import deepcopy

import numpy as np

SRC = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
ANIMAL = sys.argv[3] if len(sys.argv) > 3 else "m1"
EAR = sys.argv[4] if len(sys.argv) > 4 else "r"
sys.path.insert(0, os.path.join(SRC, "code"))
os.chdir(SRC)  # the source reads ./data/parsed/CT relative to the repo root

import cuqi  # noqa: E402
from cuqi.distribution import Gaussian, JointDistribution  # noqa: E402
from cuqi.geometry import Continuous2D  # noqa: E402
from cuqi.pde import TimeDependentLinearPDE, FD_spatial_gradient  # noqa: E402
from cuqi.model import PDEModel  # noqa: E402
from advection_diffusion_inference_utils import (  # noqa: E402
    Args, read_data_files, create_domain_geometry, create_PDE_form,
    create_prior_distribution, build_grids, create_time_steps)

assert cuqi.__version__ == "1.5.0", cuqi.__version__

# ---- STEP 1 (paper real-data diffusion-only arguments) ----------------------
args = Args()
args.animal, args.ear = ANIMAL, EAR
args.version = "realdata_diff"
args.sampler = "NUTSWithGibbs"
args.unknown_par_type = "constant"
args.unknown_par_value = [100.0]
args.data_type = "real"
args.inference_type = "heterogeneous"
args.noise_level = "0.2"
args.num_CA, args.num_ST = 5, 0
args.true_a = 0.0
args.rbc = "fromDataClip"
args.NUTS_kwargs = {"max_depth": 10, "step_size": 0.1}
args.data_grad = True
args.sampler_callback = True
args.adaptive = True
args.u0_from_data = True

# ---- STEP 2 (verbatim from advection_diffusion_inference.py) ----------------
(real_times, real_locations, real_data, real_std_data,
 diff_locations, real_data_diff, real_std_data_diff) = read_data_files(args)
real_bc_l = real_data.reshape([len(real_locations), len(real_times)])[0, :]
real_bc_l[real_bc_l < 0] = 0
real_bc_r = real_data.reshape([len(real_locations), len(real_times)])[-1, :]
real_bc_r[real_bc_r < 0] = 0
real_u0 = real_data.reshape([len(real_locations), len(real_times)])[:, 0]
real_u0[real_u0 < 0] = 0
locations = real_locations
times = real_times

# ---- STEPS 4-10 (verbatim) --------------------------------------------------
factor_L = 1.2 if args.rbc == 'zero' else 1.01
L = locations[-1] * factor_L
coarsening_factor = 5
n_grid_c = 20
grid, grid_c, grid_c_fine, h, n_grid = build_grids(L, coarsening_factor, n_grid_c)
u0 = np.interp(grid, locations, real_u0)
tau_max = 30 * 60
cfl = 5
tau = create_time_steps(h, cfl, tau_max, args.adaptive)
G_c = create_domain_geometry(grid_c, args.inference_type)
PDE_form = create_PDE_form(real_bc_l, real_bc_r, grid, grid_c, grid_c_fine,
                           n_grid, h, times, args.inference_type, u0=u0)
PDE = TimeDependentLinearPDE(PDE_form, tau, grid_sol=grid, method='backward_euler',
                             grid_obs=locations, time_obs=times,
                             observation_map=FD_spatial_gradient)
G_cont2D = Continuous2D((diff_locations, times))
A = PDEModel(PDE, range_geometry=G_cont2D, domain_geometry=G_c)

# ---- STEPS 11, 13, 15, 16 (verbatim for NUTSWithGibbs, real, data_grad) -----
x = create_prior_distribution(G_c, args.inference_type)
y = Gaussian(A(x), lambda s: 1 / s, geometry=G_cont2D)
data = real_data_diff
s = cuqi.distribution.Gamma(1.2, 5)
joint = JointDistribution(x, s, y)
posterior = joint(y=data)

n = n_grid
nt = len(tau)
nc = len(grid_c)

# ---- Fixed linear maps, probed from the source -------------------------------
Ic = np.column_stack([np.interp(grid_c_fine, grid_c, e) for e in np.eye(nc)])
Dx = -np.diag(np.ones(n), 0) + np.diag(np.ones(n - 1), 1)
vec = np.zeros(n); vec[0] = 1
Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0) / h
# Check the operator the source assembles equals -Dx' diag(Ic c) Dx.
c_test = np.linspace(300.0, 500.0, nc)
diff_op, _, ic = PDE_form(c_test, tau[3])
assert np.allclose(diff_op, -Dx.T @ np.diag(Ic @ c_test) @ Dx, rtol=0, atol=1e-12 * np.abs(diff_op).max())
assert np.array_equal(ic, u0)
# Per-step boundary values (backward Euler evaluates the source at t_{k+1}).
bl = np.array([np.interp(t, times, real_bc_l) for t in tau[1:]])
br = np.array([np.interp(t, times, real_bc_r) for t in tau[1:]])
for k, t in enumerate(tau[1:]):
    _, g, _ = PDE_form(c_test, t)
    cf = Ic @ c_test
    assert np.isclose(g[0], cf[0] / h**2 * bl[k]) and np.isclose(g[-1], cf[-1] / h**2 * br[k])
    assert np.all(g[1:-1] == 0)
dts = np.diff(tau)

# Observation map: interpolate_on_observed_domain (RectBivariateSpline) then
# FD_spatial_gradient. Probe interpolation with unit solutions, check separable.
T4 = np.zeros((len(locations), len(times), n, nt))
for i in range(n):
    for j in range(nt):
        E = np.zeros((n, nt)); E[i, j] = 1.0
        T4[:, :, i, j] = PDE.interpolate_on_observed_domain(E)
M = T4.transpose(0, 2, 1, 3).reshape(len(locations) * n, len(times) * nt)
U_, S_, Vt_ = np.linalg.svd(M)
assert S_[1] < 1e-12 * S_[0], f"observation map not separable: {S_[:3]}"
Px = (U_[:, 0] * S_[0]).reshape(len(locations), n)
Pt = Vt_[0].reshape(len(times), nt)
recon = np.einsum('li,mj->lmij', Px, Pt)
sep_err = float(np.abs(recon - T4).max() / np.abs(T4).max())
Gfd = np.zeros((len(locations) - 1, len(locations)))
for i in range(len(locations) - 1):
    Gfd[i, i] = 1.0 / (locations[i] - locations[i + 1])
    Gfd[i, i + 1] = -1.0 / (locations[i] - locations[i + 1])


# ---- Reference forward map in plain numpy (the Impulse program's algebra) ----
def forward_np(dvec):
    cf = Ic @ (dvec ** 2)
    K = -Dx.T @ np.diag(cf) @ Dx
    u = u0.copy()
    U = [u.copy()]
    for k in range(nt - 1):
        Amat = np.eye(n) - dts[k] * K
        rhs = u.copy(); rhs[0] += dts[k] * cf[0] / h**2 * bl[k]; rhs[-1] += dts[k] * cf[-1] / h**2 * br[k]
        u = np.linalg.solve(Amat, rhs)
        U.append(u.copy())
    U = np.array(U).T
    return Gfd @ (Px @ U @ Pt.T)


pts = [np.full(nc, 20.0), np.linspace(18.0, 23.0, nc), 20.0 + 2.0 * np.sin(np.arange(nc))]
refs = []
for p in pts:
    a_src = np.asarray(A(p)).reshape(len(diff_locations), len(times))
    a_np = forward_np(p)
    rel = float(np.abs(a_src - a_np).max() / np.abs(a_src).max())
    for sv in (0.05, 0.3):
        lp = float(posterior.logd(x=p, s=sv))
        refs.append({"d": p.tolist(), "s": sv, "joint_logd": lp})
    refs[-1]["forward_src"] = a_src.tolist()
    refs[-1]["forward_rel_diff_numpy_vs_source"] = rel
    print(f"forward parity numpy-vs-source rel {rel:.3e}")

out = {
    "source": {
        "paper": "Alghamdi, Mathiesen, Miyakoshi, Nedergaard, Jorgensen, Bork. JARO 27(4):555-567 (2026). doi:10.1007/s10162-026-01047-x",
        "code": "https://github.com/CUQI-DTU/Paper-Cochlear commit 912ec03b; Zenodo doi:10.5281/zenodo.15796673 (record 18503457, v1.1.0, CC-BY-4.0, zip md5 7b32c09a4925fd3bb16f4e112e801c26)",
        "cuqipy": cuqi.__version__, "numpy": np.__version__,
        "configuration": "real_data_results.ipynb diffusion-only: data_type real, inference_type heterogeneous, num_CA 5, num_ST 0, rbc fromDataClip, --data_grad --adaptive --u0_from_data, NUTSWithGibbs",
        "animal": ANIMAL, "ear": EAR,
    },
    "n": n, "nt": nt, "nc": nc, "h": h, "L": L,
    "grid": grid.tolist(), "grid_c": grid_c.tolist(), "tau": tau.tolist(), "dts": dts.tolist(),
    "locations": locations.tolist(), "times": times.tolist(),
    "Ic": Ic.tolist(), "Dx": Dx.tolist(), "bl": bl.tolist(), "br": br.tolist(), "u0": u0.tolist(),
    "Px": Px.tolist(), "Pt": Pt.tolist(), "Gfd": Gfd.tolist(), "observation_separable_max_rel_err": sep_err,
    "data": np.asarray(data).reshape(len(diff_locations), len(times)).tolist(),
    "prior_d_mean": 20.0, "prior_d_std": 5.0, "gamma_shape": 1.2, "gamma_rate": 5.0,
    "reference_points": refs,
}
with open(OUT, "w") as f:
    json.dump(out, f)
print(f"n={n} nt={nt} steps={nt-1} dts={np.unique(np.round(dts, 9))} sep_err={sep_err:.2e}")
print(f"wrote {OUT}")
