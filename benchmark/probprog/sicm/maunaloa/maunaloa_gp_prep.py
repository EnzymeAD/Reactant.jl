#!/usr/bin/env python
"""
Data preparation for the Mauna Loa CO2 GP-regression SICM (EigenLift) benchmark.

Provenance: the Scripps/NOAA Mauna Loa atmospheric CO2 record, as bundled with
statsmodels (`sm.datasets.co2`, weekly flask measurements 1958-2001). We read
the CSV directly from the installed statsmodels package (no network), so the
data is a local, versioned copy. This is the same series used in the textbook
GP example of Rasmussen & Williams (GPML 2006, Chapter 5.4.3) and the scikit-learn
GP tutorial.

Preparation (standard textbook treatment):
  1. Resample the weekly series to monthly means, drop months with no data.
  2. x = time in years since the first observation (monthly spacing preserved,
     so the 12-month seasonal period is exact).
  3. Remove a quadratic trend (least squares) capturing the accelerating secular
     CO2 rise. The residual (mean ~0) is dominated by the seasonal cycle plus
     medium-term irregularities, well modeled by a stationary kernel.
  4. K_base = fixed-shape kernel matrix (computed ONCE here, sample-invariant by
     construction), a sum of two textbook components with FIXED hyperparameters:
        k_periodic(dt) = exp(-2 sin^2(pi |dt| / P) / l_per^2), P = 1 year (season)
        k_SE(dt)       = exp(-dt^2 / (2 l_se^2))              (medium-term smooth)
     normalized so diag(K_base) = 1 (so sigma_f^2 is the marginal signal variance).

The Bayesian model (fit downstream in Impulse and NumPyro) infers ONLY the
overall signal amplitude sigma_f and observation noise sigma_n:
     y ~ MVN(0, sigma_f^2 * K_base + sigma_n^2 * I)
which is exactly the s*A + t*I form that SICM's CholeskyEigenLift targets.

Writes one JSON per subsample size N (contiguous first-N months) to the log dir.

Usage:
  /home/sbrantq/anaconda3/envs/probprog/bin/python maunaloa_gp_prep.py
"""
import os
import json
import numpy as np
import pandas as pd

OUT_DIR = os.environ.get("MAUNALOA_DATA_DIR", "/tmp/sbrantq-heatinv/maunaloa")
CSV = os.environ.get(
    "MAUNALOA_CSV",
    "/home/sbrantq/anaconda3/lib/python3.12/site-packages/statsmodels/datasets/co2/co2.csv",
)

# Fixed kernel hyperparameters (textbook, held constant across inference).
PERIOD = 1.0        # years  (annual seasonal cycle)
L_PER = 1.0         # periodic smoothness (dimensionless, R&W ExpSineSquared)
L_SE = 1.5          # years  (medium-term smooth variation)
DETREND_ORDER = 2   # quadratic secular trend removed
SUBSAMPLE_NS = [150, 250, 400, 521]


def load_monthly():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.dropna(subset=["co2"]).set_index("date")
    m = df["co2"].resample("MS").mean().dropna()
    y = m.values.astype(np.float64)
    t = (m.index - m.index[0]).days.values.astype(np.float64) / 365.25
    return t, y, m.index


def detrend(t, y, order):
    A = np.vander(t, order + 1, increasing=True)   # [1, t, t^2, ...]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    trend = A @ coef
    return y - trend, coef


def kernel_base(t):
    dt = np.abs(t[:, None] - t[None, :])
    k_per = np.exp(-2.0 * np.sin(np.pi * dt / PERIOD) ** 2 / (L_PER ** 2))
    k_se = np.exp(-(dt ** 2) / (2.0 * L_SE ** 2))
    K = k_per + k_se
    K = 0.5 * (K + K.T)                 # symmetrize against roundoff
    K = K / np.diag(K).mean()           # normalize so diag ~ 1 (exactly 1 here)
    return K


def median_heuristic(t):
    dt = np.abs(t[:, None] - t[None, :])
    return float(np.median(dt[dt > 0]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_full, y_full, idx = load_monthly()
    resid_full, coef = detrend(t_full, y_full, DETREND_ORDER)
    med = median_heuristic(t_full)
    print(f"monthly points: {len(t_full)}  span {t_full[-1]:.2f} yr "
          f"({idx.min().date()}..{idx.max().date()})")
    print(f"quad trend coef (a0,a1,a2): {coef}")
    print(f"residual std={resid_full.std():.4f} mean={resid_full.mean():.2e}")
    print(f"median-heuristic |dt| (full record) = {med:.3f} yr "
          f"(long-term scale; removed by detrend). Kernel uses fixed "
          f"l_se={L_SE} yr, l_per={L_PER}, P={PERIOD} yr.")

    summary = {}
    for N in SUBSAMPLE_NS:
        n = min(N, len(t_full))
        t = t_full[:n].copy()
        y = resid_full[:n].copy()
        K = kernel_base(t)
        evals = np.linalg.eigvalsh(K)
        attrs = dict(
            n=int(n),
            period=PERIOD, l_per=L_PER, l_se=L_SE,
            detrend_order=DETREND_ORDER,
            trend_coef=coef.tolist(),
            resid_std=float(y.std()),
            median_heuristic_yr=med,
            t_span_yr=float(t[-1] - t[0]),
            kmin_eig=float(evals.min()), kmax_eig=float(evals.max()),
            source="statsmodels sm.datasets.co2 (Scripps/NOAA Mauna Loa weekly, 1958-2001)",
        )
        out = dict(attrs=attrs, x=t.tolist(), Y=y.tolist(),
                   K_base=[row.tolist() for row in K])
        path = os.path.join(OUT_DIR, f"maunaloa_data_N{n}.json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"N={n}: K_base eig in [{evals.min():.3e}, {evals.max():.3e}], "
              f"resid_std={y.std():.3f} -> {path}")
        summary[n] = attrs
    with open(os.path.join(OUT_DIR, "prep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
