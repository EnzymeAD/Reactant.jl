#!/usr/bin/env python3
"""Cross-arm agreement at two horizons.

Pointwise (short horizon): with a shared trial seed the SICM and No-SICM arms
must take the SAME NUTS trajectory, so every draw should match to the float64
ULP floor. This holds only while the two trajectories have not yet been
separated by rounding: NUTS + dual averaging + Welford mass adaptation is
chaotic, so an O(1e-16) difference in one leapfrog step grows exponentially and
the two arms decorrelate after a few hundred iterations. The 20-iteration smoke
runs are therefore the pointwise gate.

Distributional (measured horizon): at 500 iterations the arms are two
independent-in-practice chains of the same sampler on the same target, so the
gate is that their posterior summaries agree within Monte Carlo error. This
script reports, per variable, the standardized difference of posterior means

    z = (mean_A - mean_B) / sqrt(sd_A^2/ess_A + sd_B^2/ess_B)

which is ~N(0,1) when the two arms target the same distribution, plus the ratio
of posterior sds and the PLL difference against the within-arm chain spread.

Usage: arm_agreement.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys

import arviz
import numpy as np
import xarray as xr


def run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    nw = cfg.get("num_warmup", 0)
    ds = xr.open_dataset(os.path.join(run_dir, "samples.nc")).isel(
        draw=slice(nw, None))
    metrics = xr.open_dataset(os.path.join(run_dir, "metrics.nc"))
    ppls = [str(p) for p in ds.ppl.values]
    a, b = ppls[0], ppls[1]
    print("== %s   %s (%s)" % (run_dir, cfg["model"]["class"],
                               cfg["model"].get("args")))
    print("   arms: %s vs %s ; %d post-warmup draws x %d chains"
          % (a, b, ds.sizes["draw"], ds.sizes["chain"]))
    zmax = 0.0
    for v in ds.data_vars:
        xa = ds[v].isel(ppl=0)
        xb = ds[v].isel(ppl=1)
        ea = arviz.ess(xa.to_dataset(name=str(v)))[str(v)].values
        eb = arviz.ess(xb.to_dataset(name=str(v)))[str(v)].values
        ma = xa.values.reshape(-1, *xa.values.shape[2:]).mean(0)
        mb = xb.values.reshape(-1, *xb.values.shape[2:]).mean(0)
        sa = xa.values.reshape(-1, *xa.values.shape[2:]).std(0)
        sb = xb.values.reshape(-1, *xb.values.shape[2:]).std(0)
        mcse = np.sqrt(np.atleast_1d(sa) ** 2 / np.atleast_1d(ea)
                       + np.atleast_1d(sb) ** 2 / np.atleast_1d(eb))
        z = (np.atleast_1d(ma) - np.atleast_1d(mb)) / np.where(mcse > 0, mcse, np.nan)
        sdr = np.atleast_1d(sa) / np.where(np.atleast_1d(sb) > 0,
                                           np.atleast_1d(sb), np.nan)
        zmax = max(zmax, float(np.nanmax(np.abs(z))))
        print("   %-12s |z| max %6.2f  median %5.2f   sd ratio [%.3f, %.3f]   "
              "(k=%d components)"
              % (str(v), np.nanmax(np.abs(z)), np.nanmedian(np.abs(z)),
                 np.nanmin(sdr), np.nanmax(sdr), z.size))
    pll = metrics["pll"].values
    la = [p[~np.isnan(p)][-1] for p in pll[0]]
    lb = [p[~np.isnan(p)][-1] for p in pll[1]]
    print("   PLL[-1]: %s = %s ; %s = %s ; cross-arm gap %.3f vs within-arm "
          "chain spread %.3f / %.3f"
          % (a, ["%.3f" % x for x in la], b, ["%.3f" % x for x in lb],
             abs(np.mean(la) - np.mean(lb)), abs(la[0] - la[-1]),
             abs(lb[0] - lb[-1])))
    print("   VERDICT: max |z| over all components = %.2f  (%s)"
          % (zmax, "consistent with one target distribution"
             if zmax < 4 else "CHECK: exceeds 4 MCSE"))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
