#!/usr/bin/env python3
"""New-data sanity checks on the posterior draws of the two real-data rows.

Scotland (car_spatial): the fitted spatial field should track the observed
standardized morbidity ratio. Reports Pearson/Spearman correlation between the
posterior mean of phi and log(SMR) = log(y / E), the posterior relative risk
exp(alpha + phi) against y / E, and the ranking of the extreme districts.

Radon (linear_mixed_model): the floor coefficient should be negative (first
floor reads lower than basement) and the variance components should imply real
partial pooling. Reports the posterior mean/sd of beta, sigma_u, sigma_e, and
the implied county shrinkage factor n_j sigma_u^2 / (n_j sigma_u^2 + sigma_e^2)
at the posterior mean.

Usage: sanity.py scotland <run_dir>
       sanity.py radon    <run_dir> [<n>] [<seed>]
"""
import json
import os
import sys

import numpy as np
import xarray as xr
from scipy import stats

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "pplbench", "pplbench", "models", "real_data")


def load(run_dir):
    ds = xr.open_dataset(os.path.join(run_dir, "samples.nc"))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    nw = cfg.get("num_warmup", 0)
    return ds.isel(draw=slice(nw, None)), cfg


CAR_FILES = {"scotland": "scotland_lip_cancer.json",
             "nyc_traffic": "nyc_traffic_submap.json"}


def scotland(run_dir):
    ds, cfg = load(run_dir)
    which = cfg["model"]["args"].get("dataset", "scotland")
    raw = json.load(open(os.path.join(REAL, CAR_FILES[which])))
    print("dataset:", which)
    y = np.asarray(raw["y"], float)
    E = np.asarray(raw["E"], float)
    smr = y / E
    for i, name in enumerate(str(p) for p in ds.ppl.values):
        phi = ds["phi"].isel(ppl=i).values          # [chain, draw, area]
        alpha = ds["alpha"].isel(ppl=i).values      # [chain, draw]
        sig = ds["sigma_phi"].isel(ppl=i).values
        phi_m = phi.reshape(-1, phi.shape[-1]).mean(0)
        alpha_m = alpha.reshape(-1).mean()
        rr = np.exp(alpha_m + phi_m)
        good = smr > 0
        pear = stats.pearsonr(phi_m[good], np.log(smr[good]))
        spear = stats.spearmanr(phi_m, smr)
        print("arm %s" % name)
        print("  posterior mean alpha = %.4f   mean sigma_phi = %.4f (sd %.4f)"
              % (alpha_m, np.abs(sig).mean(), np.abs(sig).std()))
        print("  corr(mean phi, log SMR)  pearson r=%.4f p=%.3g  (%d districts with y>0)"
              % (pear[0], pear[1], good.sum()))
        print("  corr(mean phi, SMR)      spearman rho=%.4f p=%.3g"
              % (spear.statistic, spear.pvalue))
        print("  corr(posterior RR, SMR)  spearman rho=%.4f"
              % stats.spearmanr(rr, smr).statistic)
        top = np.argsort(-smr)[:5]
        bot = np.argsort(smr)[:5]
        print("  top-5 SMR districts   idx=%s SMR=%s  RR=%s"
              % (list(top), np.round(smr[top], 2).tolist(), np.round(rr[top], 2).tolist()))
        print("  bottom-5 SMR districts idx=%s SMR=%s  RR=%s"
              % (list(bot), np.round(smr[bot], 2).tolist(), np.round(rr[bot], 2).tolist()))
        print("  RR range [%.3f, %.3f], SMR range [%.3f, %.3f]"
              % (rr.min(), rr.max(), smr.min(), smr.max()))


def radon(run_dir, n=250, seed=12345):
    ds, cfg = load(run_dir)
    n = int(cfg["model"]["args"].get("n", n))
    seed = int(cfg["model"].get("seed", seed))
    raw = json.load(open(os.path.join(REAL, "radon_mn.json")))
    rng = np.random.default_rng(seed)
    N = int(raw["N"])
    idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
    county = np.asarray(raw["county_idx"], np.int64)[idx]
    counties, sizes = np.unique(county, return_counts=True)
    for i, name in enumerate(str(p) for p in ds.ppl.values):
        beta = ds["beta"].isel(ppl=i).values          # [chain, draw, feature]
        su = np.abs(ds["sigma_u"].isel(ppl=i).values.reshape(-1))
        se = np.abs(ds["sigma_e"].isel(ppl=i).values.reshape(-1))
        b = beta.reshape(-1, beta.shape[-1])
        print("arm %s" % name)
        print("  beta[0] (intercept) mean %.4f sd %.4f   95%% CI [%.4f, %.4f]"
              % (b[:, 0].mean(), b[:, 0].std(),
                 np.percentile(b[:, 0], 2.5), np.percentile(b[:, 0], 97.5)))
        print("  beta[1] (floor)     mean %.4f sd %.4f   95%% CI [%.4f, %.4f]   P(beta1<0)=%.3f"
              % (b[:, 1].mean(), b[:, 1].std(),
                 np.percentile(b[:, 1], 2.5), np.percentile(b[:, 1], 97.5),
                 (b[:, 1] < 0).mean()))
        print("  sigma_u mean %.4f sd %.4f ; sigma_e mean %.4f sd %.4f ; ICC=%.4f"
              % (su.mean(), su.std(), se.mean(), se.std(),
                 su.mean() ** 2 / (su.mean() ** 2 + se.mean() ** 2)))
        lam = sizes * su.mean() ** 2 / (sizes * su.mean() ** 2 + se.mean() ** 2)
        print("  county shrinkage weight n_j su^2/(n_j su^2 + se^2): "
              "n_j=1 -> %.3f, median county (n_j=%d) -> %.3f, largest (n_j=%d) -> %.3f"
              % (lam[sizes == sizes.min()][0], int(np.median(sizes)),
                 float(np.median(lam)), sizes.max(), lam[np.argmax(sizes)]))
        print("  %d counties, sizes min/med/max = %d/%d/%d"
              % (len(counties), sizes.min(), int(np.median(sizes)), sizes.max()))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "scotland":
        scotland(sys.argv[2])
    else:
        radon(*sys.argv[2:])
