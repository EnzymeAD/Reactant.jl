#!/usr/bin/env python
"""Reference posterior from the SOURCE sampler (CUQIpy 1.5.0 NUTSWithGibbs with
finite-difference gradients, step 0.1, max depth 10) for the cochlear diffusion
row. The problem is built exactly as in cochlear_diffusion_export.py (steps 1-16
of the source driver); sampling calls the source's sample_the_posterior.
Usage: python cochlear_source_reference.py <Paper-Cochlear dir> <out.npz> [animal] [ear] [Ns] [seed]
"""

import json
import os
import sys
from copy import deepcopy

import numpy as np

SRC = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]  # output .npz for the reference samples
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


from advection_diffusion_inference_utils import sample_the_posterior  # noqa: E402
import time  # noqa: E402
Ns = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
np.random.seed(seed)  # the source driver seeds numpy with 1
args.Ns, args.Nb = Ns, 0
t0 = time.time()
samples, sampler = sample_the_posterior(args.sampler, posterior, G_c, args, callback=None)
el = time.time() - t0
xs = samples["x"].samples.T   # Ns x 21 (d)
ss = samples["s"].samples.ravel()
np.savez(OUT, d=xs, s=ss, seconds=el, Ns=Ns, seed=seed)
Dbar = float(np.mean(np.mean(xs**2, axis=0)))
print(f"source reference: Ns={Ns} seconds={el:.1f} D_bar={Dbar:.3f} sigma_mean={np.mean(ss**-0.5):.4f}")
