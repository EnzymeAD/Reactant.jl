# SICM whole-pass ablation on provenance-bearing models (2026-08-25)

Rebuild and re-measurement of the roster behind `paper/figures/sicm_ablation_bar_data.json`
(the figure that replaced `tab:sicm-ablation`). The previous roster was seven
synthetic instances; this one is four models, three of which are fitted to
published observed data, one per SICM pattern family that the figure claims.

Two arms per model, whole pass, no per-pattern knobs:

- **SICM**: the standard `:probprog` region pipeline.
- **No SICM**: the same pipeline with the `sicm` pass given zero
  rewrite+hoist iterations.

The two arms are one pass option apart. From the pass-pipeline string recorded
in the module dumps (`hlo/*_pipeline.txt`), the two pipelines are character-for-character
identical except:

```
SICM     ... inline-infer-regions, ..., sicm{gram-form-sufficient-stats=false max-iterations=10}, ..., expand-impulse{...}
No SICM  ... inline-infer-regions, ..., sicm{gram-form-sufficient-stats=false max-iterations=0},  ..., expand-impulse{...}
```

which is `Reactant.jl/src/compiler/OptimizationPasses.jl:224`
(`sicm = disable_sicm ? "sicm{max-iterations=0}" : "sicm"`),
reached from the harness flag `--disable-sicm` -> `DEBUG_PROBPROG_DISABLE_SICM`.
Logpdf-region construction still runs in both arms; only the rewrite+hoist
fixpoint is disabled.

## Roster

| # | Pattern | Model | Data | n | Julia model | Config |
|---|---|---|---|---|---|---|
| 1 | CholScale | Leroux CAR disease mapping | Scotland lip cancer, 56 districts, observed | 56 | `sicm/car_spatial.jl` | `pplbench_configs/sicm_ablation/car_spatial_scotland.json` |
| 2 | CholScale | Leroux CAR disease mapping | posteriordb NYC traffic accidents, 250-tract contiguous sub-map, observed | 250 | `sicm/car_spatial.jl` | `pplbench_configs/sicm_ablation/car_spatial_nyc.json` |
| 3 | OuterProdScale | Hierarchical MVN | synthetic (config run exactly as it already stood) | K=60, J=2 | `sicm/hierarchical_mvn.jl` | `pplbench_configs/sicm_ablation/hierarchical_mvn.json` |
| 4 | EigenLift | Marginalized linear mixed model | Minnesota radon (posteriordb `radon_mn`), fixed-seed 250-home subsample, observed | 250 | `sicm/linear_mixed_model.jl` | `pplbench_configs/sicm_ablation/lmm_radon.json` |

Data provenance for rows 1, 2 and 4 is recorded in
`pplbench/pplbench/models/real_data/README.md` (sources, citations, the exact
sub-map / subsample rules, and the model-vs-data caveat for the NYC row).

## Protocol

- Runner `./run_pplbench.sh --threads all sicm_ablation/<config>`; both arms are
  non-optional config entries, so one invocation runs both (verified on the
  smoke runs).
- 500 iterations, 200 warmup, 2 trials, seed 12345, step size 0.05, max tree
  depth 10, step-size and mass-matrix adaptation ON, for rows 1, 2, 4.
  Row 3 is the pre-existing config, unmodified: 200 iterations, 50 warmup,
  2 trials, seed 12345, step size 0.01, depth 10, adaptation on.
- CPU only (`CUDA_VISIBLE_DEVICES=""`), fully uncapped: no `taskset`, no BLAS or
  OMP caps, XLA's Eigen pool sized from the host's 192 CPUs. Observed at ~88
  cores of self-inflicted parallelism during the heavy cells.
- Statistics: **last-trial** sampling time (the final chain's infer-phase entry
  of `timing[ppl, chain, phase]` in each run's `metrics.nc`; this is the
  statistic the current figure uses) **and** the median over that cell's
  trials. Both are reported for every cell.
- Every new config was smoke-tested at 20 iterations / 10 warmup before the
  measured runs (`logs_smoke/`).

**Binary**: `oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so`,
md5 `21a6bd08b37b54d2b8a20d8924a2bbb2`, mtime 2026-08-24 14:59:21. Recorded at
the start and end of every batch and before and after every cell; it never
changed, including across the IR-dump runs afterwards. No compiler source was
edited and nothing was rebuilt.

**Machine**: AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, one NUMA
node. Julia 1.11.7, Python 3.11.14 (conda `probprog-bench`).

**Julia frontend thread count** (immaterial to the measured statistic, recorded
for completeness): the measured sampling runs inside one XLA executable whose
pool is sized from CPU affinity, not from Julia's thread count. Batches 1 and 2
launched `julia -t 32` (`inference.py`'s fallback when `JULIA_NUM_THREADS` is
unset). `run_pplbench.sh` was changed at 2026-08-25 01:46:46 so that
`--threads all` exports `JULIA_NUM_THREADS=auto`; batches 3 and 4 therefore
launched `julia -t auto` (confirmed in `logs_batch3/`, `logs_batch4/`). The
Scotland row, measured in all four batches, straddles the change and reads
5.3x / 5.3x / 5.9x / 5.5x, i.e. within its own batch-to-batch spread.

**Headline batch**: `batch4` (02:28:26 -> 03:01:35), the only batch containing
all four figure rows, and entirely on one side of the thread-flag change.
Batches 1 and 2 are complete for the three cheap rows and are reported as
cross-batch agreement. Batch 3 was killed mid-cell by an unrelated session at
02:26 and only its Scotland cell is complete; its partial NYC cell is discussed
under *Interrupted batch* below and contributes no reported number.

---

## Results

### Headline (batch4, last-trial statistic)

| Pattern | Model | Size | SICM (s) | No SICM (s) | Speedup |
|---|---|---|---|---|---|
| CholScale | Leroux CAR, Scotland lip cancer | n=56 | 1.85 | 10.10 | 5.5x |
| CholScale | Leroux CAR, NYC traffic sub-map | n=250 | 153.06 | 600.30 | 3.9x |
| OuterProdScale | Hierarchical MVN | K=60, J=2 | 3.81 | 17.53 | 4.6x |
| EigenLift | Marginalized LMM, MN radon | n=250 | 0.15 | 6.46 | 43.1x |

### Both statistics, every batch

| Model | Batch | SICM trials (s) | SICM last | SICM median | No SICM trials (s) | No SICM last | No SICM median | Speedup last | Speedup median |
|---|---|---|---|---|---|---|---|---|---|
| Leroux CAR, Scotland | batch1 | 2.067; 1.915 | 1.915 | 1.991 | 10.442; 10.061 | 10.061 | 10.251 | 5.3x | 5.1x |
| Leroux CAR, Scotland | batch2 | 2.102; 1.906 | 1.906 | 2.004 | 10.551; 10.063 | 10.063 | 10.307 | 5.3x | 5.1x |
| Leroux CAR, Scotland | batch3 | 2.101; 1.769 | 1.769 | 1.935 | 10.546; 10.500 | 10.500 | 10.523 | 5.9x | 5.4x |
| Leroux CAR, Scotland | **batch4** | 2.071; 1.846 | **1.846** | 1.959 | 10.506; 10.099 | **10.099** | 10.303 | **5.5x** | 5.3x |
| Leroux CAR, NYC | **batch4** | 151.390; 153.062 | **153.062** | 152.226 | 631.992; 600.298 | **600.298** | 616.145 | **3.9x** | 4.0x |
| Hierarchical MVN | batch1 | 4.324; 4.080 | 4.080 | 4.202 | 18.010; 16.903 | 16.903 | 17.457 | 4.1x | 4.2x |
| Hierarchical MVN | batch2 | 4.219; 3.954 | 3.954 | 4.086 | 18.325; 17.305 | 17.305 | 17.815 | 4.4x | 4.4x |
| Hierarchical MVN | **batch4** | 4.255; 3.808 | **3.808** | 4.031 | 17.828; 17.529 | **17.529** | 17.679 | **4.6x** | 4.4x |
| Marginalized LMM, radon | batch1 | 0.619; 0.152 | 0.152 | 0.385 | 6.583; 5.179 | 5.179 | 5.881 | 34.2x | 15.3x |
| Marginalized LMM, radon | batch2 | 0.551; 0.148 | 0.148 | 0.350 | 7.512; 8.048 | 8.048 | 7.780 | 54.2x | 22.2x |
| Marginalized LMM, radon | **batch4** | 0.596; 0.150 | **0.150** | 0.373 | 8.150; 6.456 | **6.456** | 7.303 | **43.1x** | 19.6x |

Reading the two statistics. The SICM cells are stable to a few percent
everywhere. The No-SICM cells are too, except the radon LMM, whose No-SICM
trials range 5.18-8.15 s across batches. That row's SICM last-trial cell is
0.150 s, so its ratio is the most sensitive of the four: 34x / 54x / 43x on the
last-trial statistic against 15x / 22x / 20x on the median. Both are reported in
the figure's provenance block; the figure itself uses the last-trial statistic,
as it did before, and 43x is that statistic's batch4 value. There is a
reproducible warm-up ramp in trial 0 of every cell (e.g. radon SICM 0.60 s then
0.15 s), which is why the established convention drops trial 0.

### Sampling efficiency (supplementary, not the ablation statistic)

Minimum effective sample size per second of sampling, batch4:

| Model | SICM n_eff/s | No SICM n_eff/s | ratio |
|---|---|---|---|
| Leroux CAR, Scotland | 187 | 36.9 | 5.1x |
| Leroux CAR, NYC | 0.225 | 0.0057 | 39x |
| Hierarchical MVN | 1.16 | 0.222 | 5.2x |
| Marginalized LMM, radon | 2448 | 63.5 | 39x |

The NYC row's 39x mixes a timing effect with a mixing accident (see the
identifiability note below) and should not be read as a compiler result; the
wall-time table is the ablation.

---

## Validation gates

All four rows pass. Evidence files are named per gate below; the full text is in
`evidence/`.

### 1. Determinism of the measurement

The same config re-run on the same build reproduces posterior draws exactly.
Comparing batch1 against batch4 for the three rows present in both, over every
variable, every draw and both arms:

| Model | max abs difference, batch1 vs batch4 |
|---|---|
| Leroux CAR, Scotland | 0 |
| Hierarchical MVN | 0 |
| Marginalized LMM, radon | 0 |

pplbench derives the trial seeds from the model seed (12345), which is fixed, so
batch-to-batch variation is wall clock only. This is also why every batch's
n_eff / Rhat / PLL rows below are identical for a given model and arm.

### 2. Chains move and mix, in BOTH arms

Detection rule inherited from the 2026-08-22 GP singular-kernel incident
(`oopsla/evidence/sicm_ablation_gp_singular_kernel_2026-08-22.md`): a chain with
one unique draw, or Rhat > 1e3, is stuck and its timing is invalid. Nothing here
resembles that: every variable of every arm has essentially as many unique draws
as there are draws, with per-component posterior sds of order 0.1-2.

| Model | Arm | n_eff min/med/max | Rhat min/med/max | min unique draws per chain (of N) | PLL[-1] per chain |
|---|---|---|---|---|---|
| Leroux CAR, Scotland | SICM | 345.1 / 566.2 / 607.4 | 0.998 / 1.001 / 1.006 | 300 (of 300) | -125.89, -125.66 |
| Leroux CAR, Scotland | No SICM | 372.3 / 459.4 / 541.5 | 1.002 / 1.005 / 1.009 | 300 (of 300) | -124.67, -126.25 |
| Leroux CAR, NYC | SICM | 34.4 / 39.1 / 273.3 | 1.000 / 1.014 / 1.037 | 300 (of 300) | -468.30, -458.81 |
| Leroux CAR, NYC | No SICM | 3.4 / 3.9 / 155.1 | 1.019 / 1.498 / 1.653 | 300 (of 300) | -469.02, -464.52 |
| Hierarchical MVN | SICM | 4.4 / 28.9 / 84.9 | 0.999 / 1.061 / 1.450 | 150 (of 150) | -165.06, -168.84 |
| Hierarchical MVN | No SICM | 3.9 / 25.7 / 122.6 | 0.997 / 1.070 / 1.511 | 150 (of 150) | -158.49, -172.46 |
| Marginalized LMM, radon | SICM | 367.2 / 516.8 / 591.1 | 0.998 / 1.003 / 1.004 | 297 (of 300) | -290.70, -290.66 |
| Marginalized LMM, radon | No SICM | 409.9 / 433.0 / 521.0 | 1.004 / 1.006 / 1.007 | 295 (of 300) | -290.67, -289.94 |

Two rows need reading rather than a threshold, and both are properties of the
model and the iteration budget, not of the ablation:

**NYC CAR: an intercept / spatial-mean ridge, not a stuck chain.** The benchmark
model is `y_i ~ Poisson(E_i exp(alpha + phi_i))` with `phi ~ MVN(0, sigma_phi^2 Q^-1)`
and no sum-to-zero constraint on `phi`, so `alpha` and `mean(phi)` are identified
only through their sum. Chain means:

| Arm | alpha (2 chains) | mean(phi) (2 chains) | alpha + mean(phi) (2 chains) |
|---|---|---|---|
| SICM | -6.863, -6.975 | -0.192, -0.086 | **-7.0548, -7.0605** |
| No SICM | -7.651, -6.339 | +0.596, -0.725 | **-7.0553, -7.0638** |

The identified sum agrees to four decimals within and across arms while the
components wander along the ridge, which is exactly what produces `alpha`
Rhat 1.65 / n_eff 3.4 in the No-SICM arm and Rhat 1.02 / n_eff 34.7 in the SICM
arm. On the identified parameterization, the log relative risk
`alpha + phi_i` (the quantity the model is about), both arms mix well:

| Row | Arm | ess min / med | Rhat min / med / max |
|---|---|---|---|
| NYC, n=250 | SICM | 137.8 / 663.9 | 0.997 / 1.002 / 1.023 |
| NYC, n=250 | No SICM | 128.6 / 622.5 | 0.997 / 1.002 / 1.029 |
| Scotland, n=56 | SICM | 451.1 / 586.1 | 0.998 / 1.001 / 1.007 |
| Scotland, n=56 | No SICM | 431.8 / 596.8 | 0.998 / 1.001 / 1.011 |

and `sigma_phi` agrees across arms (2.1395 vs 2.1445 at NYC, 0.8823 vs 0.8742 at
Scotland). The row is gated on this parameterization. The ridge is worse at
n=250 than at n=56 simply because there are more `phi` components to trade the
intercept against.

**Hierarchical MVN: a small-budget row, symmetric across arms.** 200 iterations,
50 warmup, 241 parameters and only J=2 groups per outcome. mu has Rhat up to
1.450 (SICM) / 1.511 (No SICM) and ess min 4.4 / 3.9; 45% (SICM) and 52%
(No SICM) of `mu` components exceed Rhat 1.1. The two arms are equally affected
and agree distributionally (next gate). This is the pre-existing config, run
unmodified as instructed; its numbers refresh the previous figure's
`OuterProdScale` row on the current binary (previous value 4.0 s / 15.5 s /
3.9x, now 3.81 s / 17.53 s / 4.6x).

### 3. The two arms agree

Two horizons, because a chaotic sampler decorrelates from a rounding difference.

**Short horizon, pointwise (the 20-iteration / 10-warmup smoke runs, adaptation
on).** With the shared trial seed the two arms take the same NUTS trajectory, so
every draw should match to the float64 ULP floor. Max absolute difference over
all draws:

| Model | max abs difference over all draws of all variables | same, for the PLL |
|---|---|---|
| Leroux CAR, Scotland | 6.4e-12 | 1.6e-11 |
| Leroux CAR, NYC | 2.0e-14 | 1.0e-12 |
| Marginalized LMM, radon | 6.7e-13 | 4.1e-11 |
| (phylo, see excluded section) | 5.1e-11 | 4.8e-10 |

The Hierarchical MVN row has no 20-iteration smoke of its own (it is the
pre-existing config, already validated); its arms are checked at the measured
horizon only.

**Measured horizon, distributional (500 iterations, adaptation on).** Over 500
adaptive iterations the arms' trajectories separate, so the gate is that their
posterior summaries agree within Monte Carlo error. Standardized difference of
posterior means, `z = (mean_A - mean_B) / sqrt(sd_A^2/ess_A + sd_B^2/ess_B)`,
maximized over every component of every variable:

| Model | components | max abs z | posterior sd ratio range | cross-arm PLL gap vs within-arm chain spread |
|---|---|---|---|---|
| Leroux CAR, Scotland | 58 | 1.19 | 0.985 - 1.068 | 0.32 vs 0.23 / 1.58 |
| Leroux CAR, NYC | 252 | 0.35 | 0.972 - 1.160 | 3.21 vs 9.49 / 4.50 |
| Hierarchical MVN | 241 | 1.41 | 0.821 - 1.336 | 1.48 vs 3.78 / 13.97 |
| Marginalized LMM, radon | 4 | 2.46 | 0.985 - 1.118 | 0.37 vs 0.04 / 0.73 |

Every component is within 2.5 MCSE and every cross-arm PLL gap is at or below
the within-arm chain-to-chain spread. Full text: `evidence/arm_agreement_batch4.txt`.

### 4. Conditioning

Both new CAR instances and the radon covariance are well conditioned and
factorize. This matters because the lowering drops LAPACK's `info`
(`Enzyme-JAX/src/enzyme_ad/jax/Passes/StablehloToLapack.cpp:27` `CholeskyToPotrf`
leaves `potrf#1` unused; `LowerEnzymeXLALapack.cpp:2740` `PotrfOpLowering` never
branches on it), so a failed factorization is silent and must be excluded by
checking the data directly.

| Instance | quantity | value |
|---|---|---|
| Scotland, rho=0.99 | cond(Q) | 1224 |
| Scotland, rho=0.99 | eig(Q) range | [0.01, 12.24] |
| Scotland, rho=0.99 | cond(Q_inv) after symmetrize + 1e-6 jitter | 1224, Cholesky succeeds |
| NYC sub-map, rho=0.99 | cond(Q) | 1334 |
| NYC sub-map, rho=0.99 | eig(Q) range | [0.01, 13.34] |
| NYC sub-map, rho=0.99 | cond(Q_inv) after symmetrize + 1e-6 jitter | Cholesky succeeds |
| Radon, n=250 | eig(K_re) range | [-5.6e-15, 31], rank 74 of 250 |
| Radon, n=250 | cond(sigma_u^2 K_re + sigma_e^2 I) at (0.33, 0.75) | 7.0, Cholesky succeeds |
| Radon, n=250 | same at (1.0, 0.5) / (2.0, 0.2) / (3.0, 0.1) | 125 / 3101 / 2.8e4, all succeed |
| Radon, n=250 | cond(X'X) | 7.5, X full rank |

`K_re = ZZ'` is singular on its own (rank 74 = number of counties); the model's
covariance is `sigma_u^2 K_re + sigma_e^2 I`, which is positive definite for any
`sigma_e != 0`, and the posterior keeps `sigma_e` near 0.74 with sd 0.036, far
from zero. rho was left at 0.99 for both CAR rows; no downward adjustment was
needed. Raw output: `logs_smoke/data_conditioning.log` (Scotland),
`logs_smoke/nyc_submap_build.log` (NYC, printed by the sub-map builder),
`logs_smoke/radon_conditioning.log`, and the `cond(Q)` line each CAR run emits
at data generation (`car_spatial[scotland]: n=56, edges=132, cond(Q)=1224,
sum(y)=536`; `car_spatial[nyc_traffic]: n=250, edges=630, cond(Q)=1334,
sum(y)=1353`).

### 5. New-data sanity

**Scotland.** The fitted spatial field tracks the observed standardized
morbidity ratio: Pearson r(mean phi, log SMR) = 0.965 (SICM) / 0.963 (No SICM)
over the 54 districts with y > 0; Spearman rho(posterior relative risk, SMR)
= 0.961 / 0.962. The five highest-SMR districts (SMR 6.43, 4.48, 3.67, 3.60,
3.49; these are districts 0-4, the northern high-agriculture districts of the
classic analysis) receive fitted relative risks 4.33, 4.29, 3.40, 2.43, 3.24,
and four of those five are also in the top five by fitted risk. The fit shrinks
toward the overall rate as a CAR smoother should: fitted risk spans [0.35, 4.34]
against a raw SMR span of [0.00, 6.43], and the shrinkage is strongest at the
low end, where the two zero-count districts are pulled to 0.64 and 0.65 rather
than to zero. Posterior mean alpha = 0.098 on the log scale;
sigma_phi = 0.882 (sd 0.130). Since the spatial field is lognormal with a
sizeable variance, exp(alpha) is not expected to equal the crude aggregate rate
sum(y)/sum(E) = 536/536.2 = 1.00, and no such comparison is claimed.

**NYC sub-map.** Pearson r(mean phi, log(y/E)) = 0.929 (SICM) / 0.931
(No SICM) over the 209 tracts with y > 0; Spearman rho(fitted rate, y/E) = 0.947
/ 0.947, i.e. the fitted per-tract rate reproduces the raw accident rate's
ordering across the sub-map. Posterior mean alpha = -6.92 on the log scale,
against a crude aggregate rate of sum(y)/sum(E) = 1353/983806 = 1.375e-3
(log -6.589); as at Scotland, alpha is the log-scale intercept of a lognormal
field, not an estimate of the crude rate. sigma_phi = 2.14, much larger than
Scotland's 0.88, which is consistent with the far wider exposure range
(E from 10 to 12611) and count range (y from 0 to 57) across census tracts.

**Radon.** The floor effect is negative and unambiguous: beta_floor = -0.597
(sd 0.136), 95% interval [-0.854, -0.323], P(beta_floor < 0) = 1.000, against
an OLS estimate of -0.581 on the same subsample and the textbook value of
about -0.6. Intercept (basement log radon) 1.475. Variance components
sigma_u = 0.270, sigma_e = 0.744, ICC 0.117. County partial pooling is real
and size-dependent: the implied shrinkage weight
`n_j sigma_u^2 / (n_j sigma_u^2 + sigma_e^2)` is 0.117 for a one-home county,
0.209 for the median county (n_j = 2) and 0.803 for the largest (n_j = 31), so
small counties are pulled to the grand mean and the largest is mostly its own
data. All figures are the SICM arm; the No-SICM arm agrees to the third decimal
(beta_floor -0.593, sigma_u 0.269, sigma_e 0.739).

Full text: `evidence/sanity_scotland.txt`, `evidence/sanity_nyc.txt`,
`evidence/sanity_radon.txt`.

### 6. IR evidence: which rewrite fired, and where the factorization sits

Method: one single-arm 6-iteration run per (model, arm), each into its own
output directory so the two arms' module dumps never mix
(`pplbench_configs/probe/sicm_ir_*.json`, driver `run_ir_dumps.sh`). The final
IR (`*_pre_xla_compile.mlir`, kept in `hlo/`) is classified by
`stablehlo.while` nesting depth with `analyze_ir.py`, which reuses
`ablation/pre_lowering/analyze_ir.py`'s depth walker verbatim so both ablations
are classified by identical code. Depth 0 is top level (the InitHMC gradient),
depth 1 is inside the NUTS sampling while, depth 3 is the per-leapfrog-step
body. LAPACK sites are `stablehlo.custom_call @enzymexla_compile_cpu`
classified by result signature: `(tensor<NxNxf64>, tensor<i64>)` is potrf,
`(tensor<NxNxf64>, tensor<Nxf64>, tensor<i64>)` is syevd, a single tensor result
is trsm.

**Per leapfrog step (depth 3), model-sized LAPACK only.** The
position-sized sites (58, 252, 241 and 4) are the sampler's own dense mass
matrix and are identical in both arms; they are listed separately below.

| Model | Arm | potrf(n x n) | trsm, n x n RHS | trsm, n x 1 RHS | syevd(n) |
|---|---|---|---|---|---|
| CAR Scotland, n=56 | No SICM | **2** | **4** | 8 | 0 |
| CAR Scotland, n=56 | SICM | **0** | **0** | 8 | 0 |
| CAR NYC, n=250 | No SICM | **2** | **4** | 8 | 0 |
| CAR NYC, n=250 | SICM | **0** | **0** | 8 | 0 |
| Hierarchical MVN, K=60 | No SICM | **2** | **4** | 0 | 0 |
| Hierarchical MVN, K=60 | SICM | **0** | **0** | 0 | 0 |
| Marginalized LMM, n=250 | No SICM | **2** | **4** | 8 | 0 |
| Marginalized LMM, n=250 | SICM | **0** | **0** | **0** | 0 |

**Whole program, model-sized LAPACK, by depth.**

| Model | Arm | depth 0 | depth 3 |
|---|---|---|---|
| CAR Scotland | No SICM | 2 potrf(56), 2 trsm(56x56), 6 trsm(56x1) | 2 potrf(56), 4 trsm(56x56), 8 trsm(56x1) |
| CAR Scotland | SICM | **1 potrf(56)**, 0 trsm(56x56), 6 trsm(56x1) | **0 potrf**, 0 trsm(56x56), 8 trsm(56x1) |
| CAR NYC | No SICM | 2 potrf(250), 2 trsm(250x250), 6 trsm(250x1) | 2 potrf(250), 4 trsm(250x250), 8 trsm(250x1) |
| CAR NYC | SICM | **1 potrf(250)**, 0 trsm(250x250), 6 trsm(250x1) | **0 potrf**, 0 trsm(250x250), 8 trsm(250x1) |
| Hierarchical MVN | No SICM | 2 potrf(60), 2 trsm(60x60) | 2 potrf(60), 4 trsm(60x60) |
| Hierarchical MVN | SICM | **1 potrf(60)**, 0 trsm(60x60) | **0 potrf**, 0 trsm |
| Marginalized LMM | No SICM | 2 potrf(250), 2 trsm(250x250), 6 trsm(250x1) | 2 potrf(250), 4 trsm(250x250), 8 trsm(250x1) |
| Marginalized LMM | SICM | **1 syevd(250)**, 0 potrf(250), 0 trsm(250) | **0 LAPACK at model size** |

Sampler-side sites, identical in both arms of every row (the dense
inverse-mass-matrix factorization and solve): CAR Scotland 1 potrf(58) +
1 trsm(58) at depth 0 and 1 + 1 at depth 1; CAR NYC the same at 252;
Hierarchical MVN at 241; marginalized LMM at 4.

**Which rewrite fired, per row.** The arm mechanism (`max-iterations=10` vs `0`)
is the only difference between the two IRs, so any structural difference is the
pass. Stated as what is observable in the dumps, not as pattern names inferred
from the model source.

| Row | Dominant rewrite the model matches | Observed in the final IR |
|---|---|---|
| CAR Scotland, CAR NYC | `CholeskyScaleFactorizationHLO`: `chol(sigma^2 Q_inv) -> sigma * chol(Q_inv)` | model `potrf(n)` 2 -> 0 at depth 3, and 2 -> 1 at depth 0 (the forward factor and its reverse-mode twin collapse onto one invariant factorization). All 4 matrix-RHS `trsm(n x n)` per step go to 0. The 8 vector solves `trsm(n x 1)` per step survive in both arms: their right-hand side is the sampled `phi`, so no scale rewrite can hoist them. |
| Hierarchical MVN | `CholeskyOuterProductScaleHLO`: `chol(diag(s) Omega diag(s)) -> diag(s) chol(Omega)` | model `potrf(60)` 2 -> 0 at depth 3 and 2 -> 1 at depth 0; all 4 `trsm(60 x 60)` per step go to 0, and nothing model-sized is left in the step at all, because the factor feeds a matmul (`L' eta`) rather than a solve. |
| Marginalized LMM, radon | `CholeskyEigenLiftHLO`: `chol(s K_re + t I)` answered from a hoisted eigendecomposition of `K_re` | one `syevd(250)` at depth 0 and **no model-sized LAPACK anywhere else**, in the step or in the gradient. `syevd` appears in this row's SICM arm and in none of the other seven IRs. |

**Downstream cascade, measured.** Counting model-sized elementwise work by the
same depth walk (`evidence/ir_cascade.txt`), the rewrites also thin the
per-step dense algebra, most sharply where the whole covariance construction is
absorbed:

| Model | Arm | `n x n` multiply/divide, depth 0 / depth 3 | `log` of an n-vector, depth 0 / depth 3 |
|---|---|---|---|
| CAR Scotland, n=56 | No SICM | 4 / 6 | 2 / 4 |
| CAR Scotland, n=56 | SICM | 3 / **4** | 2 / 4 |
| CAR NYC, n=250 | No SICM | 4 / 6 | 2 / 4 |
| CAR NYC, n=250 | SICM | 3 / **4** | 2 / 4 |
| Hierarchical MVN, K=60 | No SICM | 7 / 10 | 2 / 0 |
| Hierarchical MVN, K=60 | SICM | 2 / **2** | 2 / 0 |
| Marginalized LMM, n=250 | No SICM | 7 / 10 | 1 / 2 |
| Marginalized LMM, n=250 | SICM | **0 / 0** | 1 / 2 |

Two things this rules in and one it rules out. It rules in that the CAR rows
keep a residual `O(n^2)` dense construction per step (4 sites, down from 6) and
that the LMM row keeps none at all, which matches the LAPACK table. It rules
out any claim that `LogMultiplyDistributionHLO` visibly fires on these rows:
the count of `log` applied to an n-vector is unchanged by the arm in all four
models, so the log-determinant term is not observably restructured in the final
IR, whatever the pass does internally. That pattern is therefore **not** claimed
as fired here.

**What the inventory does and does not explain.** In every row SICM removes at
least two `potrf(n)` and four matrix-RHS `trsm(n)` from the leapfrog step, i.e.
all of the model's `O(n^3)` per-step LAPACK. What differs is what is *left*:

- The two CAR rows keep 8 `trsm(n x 1)` per step in both arms, because the
  density's solves act on the sampled `phi` and are irreducibly per-sample; they
  also keep 4 model-sized elementwise sites per step, down from 6. Both arms
  additionally pay the sampler's own dense mass matrix, whose size tracks the
  model (58 x 58 at Scotland, 252 x 252 at NYC).
- The LMM row keeps **nothing** model-sized in the step: EigenLift replaces the
  entire factor-and-solve chain with a hoisted `syevd` plus eigen-space algebra
  that XLA fuses. Its per-step inventory goes from 2 potrf + 4 matrix trsm +
  8 vector trsm + 10 dense elementwise sites to zero of each.

That inventory explains the *ordering* of the four rows: the row that empties
the step entirely (radon LMM, 43x) is an order of magnitude above the three that
leave per-sample work behind (3.9x-5.5x).

It does **not** explain the direction of the CAR rows' change with n. On flop
counting alone, removing `O(n^3)` work while leaving `O(n^2)` work should make
the ratio grow with n, and the measurement goes the other way: 5.5x at n=56 and
3.9x at n=250. The per-step wall clock is therefore not flop-proportional at
these sizes, which is unsurprising when a 250 x 250 `dpotrf` is a blocked,
multithreaded LAPACK call on a 96-core host while the surviving vector solves
and the sampler's own bookkeeping are latency-bound; but the wall clock was not
decomposed per kernel, so that is a plausible mechanism and not a measured one.
The measured statement is the one in the table: at n=56 the hoist is worth 5.5x
and at n=250 it is worth 3.9x. **This contradicts a bigger-n-bigger-win
expectation and is reported as measured, not tuned.** It is not a measurement
artifact: the two NYC SICM trials are 151.4 and 153.1 s and the two No-SICM
trials 632.0 and 600.3 s, a few percent apart, and the arms agree
distributionally to 0.35 MCSE.

The small-n note for the Scotland row is therefore not "n=56 is too small to see
anything". At n=56 the hoisted factorization is already worth 5.5x, and the
larger CAR instance does not improve on it.

Reproduce the IR evidence:

```
cd oopsla/Reactant.jl/benchmark/probprog/ablation/sicm_roster
python3 analyze_ir.py lapack hlo/car_spatial_scotland_{sicm,nosicm}.mlir
python3 analyze_ir.py lapack hlo/car_spatial_nyc_{sicm,nosicm}.mlir
python3 analyze_ir.py lapack hlo/lmm_radon_{sicm,nosicm}.mlir
python3 analyze_ir.py lapack hlo/hierarchical_mvn_{sicm,nosicm}.mlir
```

Exact result signatures (which separate matrix-RHS from vector-RHS solves) are
in `evidence/ir_shapes.txt`; the depth-classified counts are in
`evidence/ir_lapack.txt`; the elementwise cascade counts are in
`evidence/ir_cascade.txt`, produced by `python3 analyze_cascade.py`.

---

## Machine conditions

md5 of the binary before and after every cell, and 1-minute load average before
and after every cell. All four batches ran on the same binary; it never changed.

| Batch | md5 start | md5 end | load1 at start | load1 at end | window |
|---|---|---|---|---|---|
| batch1 | 21a6bd08 | 21a6bd08 | 3.94 | 71.81 | 01:03:43 -> 01:27:03 |
| batch2 | 21a6bd08 | 21a6bd08 | 5.39 | 72.32 | 01:30:19 -> 01:53:45 |
| batch3 | 21a6bd08 | (killed) | 7.12 | - | 01:58:52 -> killed 02:26 |
| batch4 | 21a6bd08 | 21a6bd08 | 21.81 | 27.16 | 02:28:26 -> 03:01:35 |

| Batch | Cell | load1 before | load1 after | window | exit |
|---|---|---|---|---|---|
| batch4 | car_spatial_scotland | 21.81 | 23.63 | 02:28:34 -> 02:30:37 | 0 |
| batch4 | car_spatial_nyc | 21.98 | 91.45 | 02:30:45 -> 02:57:59 | 0 |
| batch4 | lmm_radon | 84.36 | 41.46 | 02:58:07 -> 02:59:26 | 0 |
| batch4 | hierarchical_mvn | 35.70 | 36.84 | 02:59:35 -> 03:01:27 | 0 |

The high load figures are self-inflicted: the uncapped ablation regime lets
XLA's Eigen pool size itself from all 192 CPUs, and the process was observed at
about 88 cores during the heavy cells. External load was 3-4 cores throughout
(one `enzymexlamlir-opt` and three long-running `julia` processes belonging to
other users), i.e. under 2% of the machine, and did not change over the
session. `pgrep -af taskset` found no pinned sibling measurement at any batch
start. Full per-cell records: `logs_batch*/batch_meta.txt`.

---

## Measured but excluded from the figure (motivating-example overlap)

The phylogenetic Poisson regression on the brms phylogenetics vignette data
(200 species, real tree, `pplbench_configs/sicm_ablation/phylo_pois.json`,
`motivating/phylo_pois.jl`) was built and measured as a CholScale row before the
roster was revised. It is **excluded from the figure**: it is the paper's
section-2 motivating example, which was deliberately removed from the evaluation
comparison on 2026-08-01 and must not reappear as an ablation row. The
measurements are recorded here rather than discarded.

| Batch | SICM trials (s) | SICM last | SICM median | No SICM trials (s) | No SICM last | No SICM median | Speedup last | Speedup median |
|---|---|---|---|---|---|---|---|---|
| batch1 | 19.206; 20.354 | 20.354 | 19.780 | 459.085; 445.756 | 445.756 | 452.420 | 21.9x | 22.9x |
| batch2 | 19.204; 20.914 | 20.914 | 20.059 | 466.542; 444.516 | 444.516 | 455.529 | 21.3x | 22.7x |

Gates on those two runs: chains move in both arms (300 unique draws per chain
for every scalar); SICM n_eff 10.7 / 69.0 / 161.9 with Rhat max 1.159, No SICM
n_eff 7.1 / 68.1 / 308.9 with Rhat max 1.228; pointwise arm agreement at the
20-iteration horizon 5.1e-11 (draws) and 4.8e-10 (PLL); distributional agreement
at 500 iterations max |z| = 2.09 over 404 components. The config file remains in
`pplbench_configs/sicm_ablation/phylo_pois.json` but is not in `run_batch.sh`'s
cell list. No IR dump was taken for it under this roster.

## Interrupted batch (batch3)

Batch3 was killed at about 02:26 by an unrelated session pruning background
tasks. Its Scotland cell had completed (reported above). Its NYC cell had
completed the SICM arm (151.4 / 153.0 s, PLL -468.3031 / -458.8098, both
recorded in `logs_batch3/car_spatial_nyc.log`) and was inside the No-SICM arm
when the process died, so that cell has no output directory and no `metrics.nc`.
A No-SICM trial-0 time of 638.12 s for that cell was reported from the live
terminal by the session that killed it; **that line is not present in my copy of
the log** (which ends mid-line at 02:07:15 at the No-SICM server startup), so it
is recorded here as second-hand and is not used for any reported number. It is
consistent with batch4's fresh 631.99 / 600.30 s. Batch3's `lmm_radon` and
`hierarchical_mvn` cells never ran. The whole NYC row was re-measured from
scratch in batch4.

---

## Files

Created or edited for this measurement (all paths relative to
`oopsla/Reactant.jl/benchmark/probprog/`):

**Data and models**
- `pplbench/pplbench/models/real_data/nyc_traffic_submap.json` (new): the
  250-tract NYC sub-map, with an embedded `_provenance` string.
- `pplbench/pplbench/models/real_data/README.md` (new): provenance for all three
  real data sets, the sub-map / subsample policy, and the NYC model-vs-data
  caveat.
- `pplbench/pplbench/models/car_spatial.py` (edited): added a `dataset`
  parameter to `generate_data` with a real-data branch serving `"scotland"` and
  `"nyc_traffic"`; the synthetic branch and the `Dataset` schema are unchanged.
- `pplbench/pplbench/models/linear_mixed_model.py`: reviewed, **not modified**.
  The pre-existing `dataset="radon_mn"` branch is correct: the subsample is a
  pure function of the config seed and n, counties are re-indexed to those
  present so `Z` has no empty column, `attrs` reports `n`, `k = X.shape[1] = 2`
  and `q = 74` (the count actually used). `generate_init_params` reads `k` from
  those attrs, giving `beta` of length 2, which matches the Julia model's
  `position_size = k + 2 = 4`. `sicm/linear_mixed_model.jl` reads `attrs["q"]`
  but never uses it for a shape, so no particular `q` is assumed.

**Configs**
- `pplbench_configs/sicm_ablation/car_spatial_scotland.json` (new)
- `pplbench_configs/sicm_ablation/car_spatial_nyc.json` (new)
- `pplbench_configs/sicm_ablation/lmm_radon.json` (new)
- `pplbench_configs/sicm_ablation/phylo_pois.json` (new; measured, excluded)
- `pplbench_configs/sicm_ablation/hierarchical_mvn.json`: **not modified**, run
  exactly as it stood.
- `pplbench_configs/probe/sicm_smoke_*.json` (4 new): 20-iteration smoke configs.
- `pplbench_configs/probe/sicm_ir_*.json` (8 new): 6-iteration single-arm IR
  configs.

**This measurement's tree** (`ablation/sicm_roster/`)
- `sicm_roster_RESULTS_2026-08-25.md` (this file)
- `sicm_roster_RAW_2026-08-25.json`: per-batch, per-cell trial times, n_eff,
  Rhat, per-variable chain movement, cross-arm agreement, batch metadata, and
  the cross-batch summary.
- `run_batch.sh`, `run_smoke.sh`, `run_ir_dumps.sh`: the drivers.
- `build_nyc_submap.py`: the deterministic BFS sub-map builder.
- `postcheck.py`, `arm_agreement.py`, `sanity.py`, `analyze_ir.py`,
  `analyze_cascade.py`, `aggregate.py`, `report_tables.py`: the gates and the
  extraction.
- `hlo/`: the eight final-IR dumps and their pass-pipeline headers.
- `evidence/`: the gate outputs quoted above.
- `logs_batch1/` ... `logs_batch4/`, `logs_ir/`, `logs_smoke/`: run logs and
  `batch_meta.txt`. Note `benchmark/probprog/.gitignore` excludes `*.log`, so
  the logs are on disk but not picked up by `git add`; `batch_meta.txt`, the
  JSON, the MLIR dumps and `evidence/` are.
- `build_diamonds.py`, `run_diamonds.sh`, `logs_diamonds/`: **not mine.** A
  sibling session placed its files in this directory; they are untouched.

**Paper**
- `paper/figures/sicm_ablation_bar_data.json`: rows replaced with the four rows
  above, provenance block rewritten. Arm keys, order and Okabe-Ito colours
  unchanged.
- `paper/figures/sicm_ablation_bar_log.pdf`: regenerated.
- `paper/figures/gen_sicm_ablation_bar.py`: axis limits, figure height, the
  speedup column's x position, and the sub-second value-label format.
  No other paper file was touched.

## Repro

```
cd oopsla/Reactant.jl/benchmark/probprog
bash ablation/sicm_roster/run_smoke.sh              # 20-iteration smoke, all rows
bash ablation/sicm_roster/run_batch.sh <tag>        # the four measured cells
bash ablation/sicm_roster/run_ir_dumps.sh           # the eight IR dumps
python3 ablation/sicm_roster/aggregate.py out.json ablation/sicm_roster/logs_batch4
python3 ablation/sicm_roster/report_tables.py out.json batch4
```

or a single cell:

```
CUDA_VISIBLE_DEVICES="" ./run_pplbench.sh --threads all sicm_ablation/car_spatial_nyc
```

---
---

# Roster extension: TriSolveScale and DotScale on real data (2026-08-25, later)

User-approved extension of the roster from four rows to six, restoring the
`TriSolveScale` and `DotScale` pattern families to the figure on
provenance-bearing models instead of the retired synthetic "Scaled Solve" and
"Multi-Sensor" instances. Same binary, `21a6bd08b37b54d2b8a20d8924a2bbb2`,
verified before and after every cell of this extension, so these rows join the
same single-build set as rows 1-5.

| # | Pattern | Model | Data | n |
|---|---|---|---|---|
| 6 | TriSolveScale | Phylogenetic generalized least squares (PGLS) | caper `shorebird`, 71 species, observed | 71 |
| 7 | DotScale | Bayesian spatial-lag (SAR) | see below: Boston n=506 attempted and abandoned, Columbus n=49 shipped | 49 |

Both rows were taken **dump-first**: the IR was dumped and the pattern's firing
verified before any timed measurement, on the standing instruction that a
blocking shape is a reportable coverage finding rather than something to design
around.

## Coverage finding 1: the diagonal-extraction spelling blocks CholeskyEigenLift

**Status: confirmed, cause isolated, proof by construction.**

`CholeskyEigenLiftHLO` accepts a Cholesky factor whose users are triangular
solves or a diagonal extraction, and `isDiagonalExtraction`
(`Enzyme-JAX/src/enzyme_ad/jax/Passes/SampleInvariantCodeMotion.cpp:2004`)
recognises exactly two spellings of that extraction:

```
Pattern 1: multiply(L, I)
Pattern 2: dot_general(L, I, batching=[1]x[1], contracting=[0]x[0])
```

Reactant's `LinearAlgebra.diag(L)` traces to neither. In the SAR model's final
IR it appears as

```
%71 = stablehlo.transpose %70, dims = [1, 0] : tensor<506x506xf64>
%72 = stablehlo.reshape %71 : (tensor<506x506xf64>) -> tensor<256036xf64>
%74 = stablehlo.slice %73 [0:256036:507] : (tensor<256036xf64>) -> tensor<506xf64>
```

a strided slice. For a **log-determinant-only** use of the factor, which is the
canonical case for a SAR or a CAR precision, that slice is the factor's ONLY
user, so `checkPreconditions` finds `fullSolves`, `singleSolves` and
`diagExtractions` all empty and returns failure. The `O(n^3)` Cholesky then
stays inside the leapfrog step in both arms.

Proof by construction. A diagnostic variant identical except that the diagonal
is spelled `sum(L .* I; dims=1)` (i.e. `multiply(L, I)`) flips the outcome
completely:

| SAR n=506, SICM arm | potrf(506) depth 0 / depth 3 | trsm(506) depth 0 / depth 3 | syevd(506) |
|---|---|---|---|
| naive `LinearAlgebra.diag(L)` | 1 / **2** | 2 / **4** | **none** |
| diagnostic `multiply(L, I)` | 1 / **0** | 2 / **0** | **1 at depth 0** |

So the blocker is the diagonal spelling alone; nothing else about the model
stands in the way. Artifacts: `hlo/sar_boston_sicm.mlir`,
`hlo/sar_boston_diagmul_sicm.mlir`, `evidence/ir_lapack_row67.txt`. The
diagnostic model is `sicm/sar_lag_diagmul.jl`, kept and clearly marked
DIAGNOSTIC VARIANT -- NOT a measured model.

The fix would be a few lines in `isDiagonalExtraction` to recognise the
strided-slice form. No compiler source was touched here.

## Coverage finding 2: the recognized spelling fires but is numerically broken

**Status: observed and reproducible; cause NOT isolated. Recorded for future
attention, not diagnosed.**

The same diagnostic variant that makes the pattern fire produces a SICM arm
whose posterior is wrong:

| Arm | rho_raw unique draws / chain | PLL[-1] per chain |
|---|---|---|
| SICM (pattern fires) | **1, 1** (frozen) | **-9402.86, -14654.93** |
| No SICM | 10, 9 (moving) | **+236.24, +238.25** |

Cross-arm disagreement is O(1) (rho_raw 1.49, beta 1.18, sigma 0.95), and the
No-SICM arm is the one that agrees with the reference fit. What has been ruled
out: the log-determinant substitution the pattern uses is exact, checked
directly against this data's matrices,

```
rho=+0.0000  sum log diag(chol A) =  0.0000000000   sum log sqrt(1 - rho lambda) =  0.0000000000  diff 0
rho=+0.2500  ... -2.0615911159  ... -2.0615911159   diff  8.9e-16
rho=+0.4856  ... -8.5513381789  ... -8.5513381789   diff -1.8e-15
rho=-0.5000  ... -7.5243348236  ... -7.5243348236   diff -1.8e-15
```

and the rank-2 replacement does re-mask with the identity
(`SampleInvariantCodeMotion.cpp:2352-2371`), so neither the determinant value
nor an obvious missing mask explains it. This matters because the recognised
spelling is reachable from an ordinary `sum(L .* I; dims=1)` idiom, and the
failure is silent: it surfaces only as a frozen chain, which the 2026-08-22
detection rule catches by luck rather than by design.

## Row 7: the Boston n=506 attempt, abandoned

Recorded because the attempt consumed machine time and produced evidence, not
because it produced a number. **No ratio is derived from it.**

- Data built and validated: `real_data/boston_sar.json`, 506 tracts, 1076
  edges, the Harrison and Rubinfeld hedonic specification, dual weight matrices
  with the determinant identity checked to 1.7e-13, profile-likelihood
  `rho_hat = 0.4856` against the 0.45-0.50 that spdep's Boston `lagsarlm`
  examples report.
- The model is correct: the SICM arm ran cleanly, trials 58.57 s and 51.80 s,
  `PLL[-1] = +257.82` against a profile optimum of about +264.
- The No-SICM arm was killed after more than 40 minutes on trial 0 alone.
- Why the smoke did not catch it: at 20 iterations with 10 warmup the NUTS
  trees are shallow and the two arms read 0.78 s and 0.82 s. The gap only opens
  at 500/200, when the trees deepen. The pre-authorised "fall back to Columbus
  if n=506 is impractical" trigger therefore fired one stage later than
  intended.
- Contamination: the cell ran inside a load1 130-160 window created by an
  external user's julia and a bazel build. A ratio whose slow arm spans 80
  contaminated minutes against a fast arm of two quiet-ish minutes is not a
  robust quantity, whatever the back-to-back-arms principle gives for short
  cells.
- **Unreconciled wall-versus-IR inconsistency, the substantive reason the row
  was not shippable.** Dump-first says the `O(n^3)` factorization sits at depth
  3 in BOTH arms, and DotScale removed only six `O(n^2)` `dot_general` sites,
  so a greater-than-35x wall ratio cannot be explained by the `W y` hoist.

  The first hypothesis was that most of the gap was generic
  `hoistSampleInvariantOps` rather than the DotScale pattern, on the reasoning
  that `max-iterations=0` disables the whole rewrite-and-hoist fixpoint.
  **That hypothesis is refuted** by the dumps already taken
  (`evidence/ir_boston_hoisting.txt`): the two arms' final IRs are nearly
  identical in size and in how much sits inside the loop.

  | Boston arm | total ops | ops at depth >= 1 | ops at depth 3 | 506x506 ops at depth >= 1 |
  |---|---|---|---|---|
  | SICM | 4318 | 3252 | 920 | 38 |
  | No SICM | 4326 | 3258 | 926 | 48 |

  The entire structural difference is those 10 sites, which are exactly the 6
  `dot_general` and 4 `multiply` that DotScale removed. Nothing is being
  rebuilt in-loop in the No-SICM arm.

  That leaves a **sampler-trajectory** explanation as the leading account: the
  two arms decorrelate under adaptation (they always do, see gate 3), and the
  No-SICM arm appears to have adapted into a regime with far deeper NUTS trees,
  i.e. many more leapfrog steps per iteration at essentially the same per-step
  cost. The 20-iteration smoke supports this directly: with shallow trees and
  almost no adaptation the two arms read 0.78 s and 0.82 s, a 1.05x ratio, and
  the gap only opens once adaptation has had 200 warmup iterations to separate
  them. If that is what happened, the Boston "ratio" would have been measuring
  a difference in trajectory length, not a difference in compiled per-step cost,
  and would not have been a compiler result at all.

  **Methodological point this yields.** The discriminator that caught it is
  "wall and IR must agree on a mechanism". Rows 1-5 and row 6 all pass it: the
  large ratios are accompanied by an IR change that removes the dominant
  per-step kernel. Boston is the only cell in this campaign where they
  disagreed, and it is the only cell that was discarded. Any future row whose
  speedup is not matched by a corresponding structural change in the leapfrog
  body should be treated the same way.

Artifacts kept: `logs_row67a/sar_boston.log`, `logs_row67a/batch_meta.txt`
(per-cell load record), `hlo/sar_boston_{sicm,nosicm}.mlir`.

## Note for future work: the CAR rows fix rho, and need not

Not measured; recorded as the un-hacked variant. The two Leroux CAR rows
(rows 1 and 2) hold the spatial mixing parameter `rho` FIXED at 0.99, following
the empirical-Bayes practice common in routine disease-mapping surveillance.
That is a modelling choice made partly so that `Q(rho) = rho (D - W) + (1 - rho) I`
is sample-invariant and its inverse can be precomputed as data, which is what
makes those rows `CholScale` instances. Sampling `rho` instead removes the fix
entirely: `Q(rho)` becomes `rho * M + (1 - rho) * I` with `M = D - W` invariant,
which is exactly the `CholeskyEigenLift` shape `s*A + t*I`. A sampled-rho CAR
would therefore be a stronger and less arranged instance, moving those rows from
CholScale to EigenLift and removing the fixed-rho caveat from the model
description. It is blocked today by coverage finding 1 above, since a CAR
precision's log-determinant is exactly the log-determinant-only use of the
factor that the diagonal-extraction matcher does not recognise.

## Measurements: rows 6 and 7

All from the quiet window of 2026-08-25 14:19-14:34, batch `row67quiet`, binary
`21a6bd08b37b54d2b8a20d8924a2bbb2` verified before and after every cell, load1
6.1 at window start (the pre-lowering campaign had cleared at 14:17:24).

### Row 6, TriSolveScale: PGLS on the caper shorebird data (SHIPPED)

| Statistic | SICM ON | SICM OFF | ratio |
|---|---|---|---|
| trials (s) | 0.568; 0.043 | 1.326; 0.776 | |
| last-trial (figure statistic) | **0.043** | **0.776** | **18.1x** |
| trial median | 0.306 | 1.051 | 3.4x |
| matched trajectories (adaptation off) | 0.03 | 0.31 | **10.3x** |

Gates, all pass. n_eff 494/536/572 (SICM ON) and 492/497/534 (SICM OFF);
Rhat <= 1.006 and <= 1.002; every variable moves (594-598 unique draws of 300 for
`beta`, 297-299 for `sigma`); PLL[-1] +34.70 / +34.53 against +34.72 / +34.44.
Cross-arm distributional agreement max |z| = 0.68 over all components, PLL gap
0.037 against a within-arm chain spread of 0.18-0.28. Conditioning: cond(C) =
1099, Cholesky succeeds; the tree is ultrametric so diag(C) = 1 exactly.

Literature sanity: posterior `beta` = [-0.4178, +0.6762] (SICM ON) and
[-0.4279, +0.6764] (SICM OFF) against the GLS reference [-0.4218, +0.6756] with
standard errors [0.2278, 0.0348], i.e. z = 0.02 on both coefficients in both
arms. P(slope > 0) = 1.000. The slope sits on the classic allometric exponent
near 0.68 for egg mass on body mass.

IR evidence, per leapfrog step (depth 3), model-sized LAPACK:

| Arm | potrf(71) | trsm(71) |
|---|---|---|
| SICM OFF | 2 | 12 |
| SICM ON | **0** | **0** |

and at depth 0 the SICM arm keeps exactly 1 potrf(71) and 2 trsm(71): the
hoisted `chol(C)` and the two hoisted whitening solves `L_C \ y` (n x 1) and
`L_C \ X` (n x k). This is the signature the row was built to test: because the
density is written in whitened-GLS form, BOTH solve right-hand sides are data,
so the solves hoist rather than merely the factorization. Contrast the CAR rows,
where the solve right-hand side is the sampled field `phi` and 8 vector solves
per step survive in both arms.

Caveat recorded: the whitened-Cholesky spelling is this benchmark's choice. It
is one of the two standard GLS spellings and the one Stan's
`multi_normal_cholesky` and brms use, but caper's own `pgls` instead forms the
explicit inverse `iV <- solve(V)` (`caper/R/pgls.R:383-393`). The model fitted
is the vignette's; the spelling of its density is ours.

### Row 7, DotScale: Bayesian spatial-lag SAR (NOT SHIPPED)

**The row is not broken. It is not a DotScale result while EigenLift is
blocked.** Measured on Columbus (n=49, Anselin 1988) after Boston (n=506) proved
impractical:

| Statistic | SICM ON | SICM OFF | ratio |
|---|---|---|---|
| trials (s) | 0.781; 0.289 | 16.138; 11.763 | |
| last-trial | 0.289 | 11.763 | 40.7x |
| trial median | 0.535 | 13.950 | 26.1x |
| **matched trajectories (adaptation off)** | **0.24** | **0.26** | **1.08x** |

Its own gates pass: n_eff 510/678/730 and 162/210/292, Rhat <= 1.005 and
<= 1.030, PLL[-1] -47.96 / -47.97 against -48.02 / -47.79. The data and model are
sound: profile-likelihood `rho_hat` = 0.4025 matching the 0.43 that spdep's
Columbus `lagsarlm` example reports, with INC and HOVAL both negative, the
classic result.

Why it is not plotted. Two independent reasons, and either alone would be
enough:

1. **EigenLift is blocked** by coverage finding 1 above, so the model's dominant
   per-step cost, the log-determinant's `O(n^3)` Cholesky, stays inside the
   leapfrog step in BOTH arms. What DotScale removes is 6 `n x n` `dot_general`
   sites and 2 multiplies per step. The IRs confirm it: both arms carry
   2 potrf(49) + 4 trsm(49) at depth 3, total ops 4318 against 4326.
2. **The adaptation-ON ratio is trajectory divergence, not compiled cost.**
   40.7x collapses to 1.08x once both arms are made to take bit-identical
   trajectories. 1.08x is what the IR predicts, and it is the honest per-step
   number.

Neither number belongs in the figure. 40.7x is refuted. Plotting 1.08x would
mislead in the other direction, reading as "DotScale is worthless" when what it
actually measures is a half-blocked cascade. The row can return once the
diagonal-extraction matcher recognises the strided-slice spelling, which is
post-freeze compiler work.

This also **retroactively explains Boston**. Same model, same blocked pattern,
same identical-IR signature, same inflated adaptation-ON ratio. Boston was never
an n=506 problem; killing it was correct for a reason that only became visible
at n=49, where the same experiment is cheap enough to run.

## Campaign-wide finding: adaptation-ON ratios need a matched-trajectory check

The figure's statistic is adaptation-ON last-trial. Under it the two arms
decorrelate (gate 3), so they can end up taking different NUTS trajectory
lengths, and a wall-clock ratio can then reflect trajectory length rather than
compiled per-step cost. Re-running a row with adaptation off makes both arms
take bit-identical trajectories from the same seed, which isolates per-step
cost. Every cheap row was audited:

| Row | adaptation ON | matched trajectories | verdict |
|---|---|---|---|
| Lip Cancer CAR, n=56 | 5.5x | 5.7x | confirmed |
| Shorebird PGLS, n=71 | 18.1x | 10.3x | confirmed |
| Columbus SAR, n=49 | 40.7x | **1.08x** | **refuted** |
| Hier. MVN, K=60 | 4.6x | 5.0x | confirmed |
| Radon LMM, n=250 | 43.1x | 34.2x | confirmed |

Four of five confirm within noise, and each of those four is corroborated by IR
that removes the dominant per-step kernel. The one that fails is the one whose
IR showed no such removal. This is the wall-and-IR-must-agree rule made
quantitative, and it is the reason the shipped figure's numbers can be trusted.
NYC (expensive SICM arm) and Diamonds (another session's row) were not audited.

Raw logs: `logs_row67quiet/`, configs `pplbench_configs/probe/sicm_fixedstep_*.json`.

**Superseded by a better protocol, already scheduled.** The user has since
designed a frozen-adapted-state protocol for the pre-lowering figure: warm up
once with adaptation on, save the adapted state (step size, inverse mass matrix,
post-warmup position), then measure both arms adaptation-off from that shared
state with matched seeds. That keeps the realistic adapted regime AND gives
bit-identical trajectories, and it should also collapse the last-trial versus
median divergence seen in the subsecond rows, since that gap is largely trial-0
warm-up asymmetry. When the plumbing lands, this figure migrates to it and every
row is re-measured, NYC and Diamonds included. The matched-trajectory audit
above is the preview of what to expect. No duplicate plumbing was built here.

## Files added by the roster extension

- `pplbench/pplbench/models/real_data/shorebird_pgls.json`, `boston_sar.json`,
  `columbus_sar.json`, all with embedded `_provenance`; `real_data/README.md`
  extended with all three, including the dual weight-matrix rationale and the
  note that the PGLS density's whitened spelling is ours and not caper's.
- `sicm/pgls.jl`, `sicm/sar_lag.jl`, `sicm/sar_lag_diagmul.jl` (diagnostic only).
- `pplbench/pplbench/models/pgls.py`, `sar_lag.py`, `sar_lag_diagmul.py`;
  `ppls/impulse/pgls.py`, `sar_lag.py`, `sar_lag_diagmul.py`.
- Configs: `sicm_ablation/{pgls,sar_boston,sar_columbus}.json`, plus
  `probe/sicm_{smoke,ir,fixedstep}_*` for the smoke, dump-first and
  matched-trajectory runs.
- `build_shorebird_pgls.py`, `build_boston_sar.py`, `build_columbus_sar.py`,
  `run_row67_quiet.sh`.
- `hlo/`: `pgls_{sicm,nosicm}.mlir`, `sar_boston_{sicm,nosicm}.mlir`,
  `sar_boston_diagmul_sicm.mlir`, `sar_columbus_{sicm,nosicm}.mlir` and their
  pipeline headers.
- `evidence/ir_lapack_row67.txt`, `ir_lapack_columbus.txt`,
  `ir_boston_hoisting.txt`.
- Paper: `figures/sicm_ablation_bar_data.json` (six rows, SICM ON / SICM OFF
  labels), `figures/sicm_ablation_bar_log.pdf`, `figures/gen_sicm_ablation_bar.py`
  (height 2.75, xlim 0.02-1.2e4, speedup column 1.05e4),
  `figures/sicm_ablation.tex` (caption arm wording, stale roster note cleared).

## Runtime-path fix verified (QR symbols), 2026-08-25

Standing instruction recorded: where LAPACK lacks a lowering or runtime path
that a measured row needs, fix the path rather than only working around it.

The one such gap in this campaign was the QR runtime symbols. The
`GaussianSufficientStats` QR form emits `enzymexla.linalg.qr`, which lowers to
`enzymexla_lapack_{d,s}geqrf_` and `_orgqr_`, and Reactant's
`TracedLinearAlgebra.__init__` registered neither, so the diamonds row was
executable on CPU only via a local registration in its model file. Those four
entries are now permanent in `Reactant.jl/src/stdlibs/LinearAlgebra.jl`, in a QR
block just after the getrf entries. It is pure Julia, so **no rebuild**: the
binary is still `21a6bd08b37b54d2b8a20d8924a2bbb2`.

Verification performed here, end to end rather than by inspection: the local
`register_qr_lapack_symbols!()` call in `sicm/diamonds_regression.jl` was
commented out, and the diamonds smoke re-run. It compiles and both arms run
(`probe/sicm_smoke_diamonds`, run dir `outputs/2026-08-25_14:38:18`, trials
0.54 / 0.01 s and 0.58 / 0.01 s), with no "Symbols not found". The local
function is retained, unused, so the workaround can be restored in one line if
the permanent registration regresses. Log:
`logs_row67quiet/smoke_diamonds_permfix.log`.

The remaining known path gap is the diagonal-extraction matcher of coverage
finding 1. That one is compiler work and needs a rebuild, so it stays
post-freeze under the single-build discipline.

## Pending: migration to the frozen-adapted-state protocol

Scheduled, not done here, and no plumbing was built for it (the pre-lowering
session owns that work). The protocol: warm up once with adaptation on, save the
adapted state (step size, inverse mass matrix, post-warmup position), then
measure both arms adaptation-off from that shared state with matched seeds. It
keeps the realistic adapted regime while giving bit-identical trajectories, and
it should also collapse the last-trial versus median divergence in the subsecond
rows, since that gap is largely trial-0 warm-up asymmetry.

**Verification required of THIS figure before any migrated number is reported**,
per user directive, and not inherited from the pre-lowering session's own
checklist. Their verification does not transfer, because these models differ in
position layout, dense-mass dimensionality and constrained supports, which is
exactly where injection bugs hide. On receiving the plumbing, at smoke scale, on
at least one model from this roster:

1. **Injection actually consumed**: a perturbation positive control, i.e.
   injecting a deliberately wrong step size or inverse mass must visibly change
   the run.
2. **Mass-matrix shape and convention round-trip**: saved and re-injected
   inverse mass must reproduce itself, checked against the dense
   position-sized matrices these rows use (58, 252, 241, 3, 5).
3. **Arms bit-identical from shared state with matched seeds**: the
   identical-accepts gate.
4. **Regime equivalence**: the injected state must reproduce the continuous
   run's post-warmup phase, not some other regime.

The constrained-support rows are the ones to prioritize for check 2, since
`sigma`, `sigma_phi` and `sigma_u`/`sigma_e` are all HalfNormal and the position
vector is in unconstrained space. Only after these pass does the migration batch
run, and the result is recorded here as a gate like the others.

---
---

# SUPERSEDING: frozen-adapted-state re-measurement (2026-08-25, evening)

**Every number in the shipped figure now comes from this section.** The
adaptation-ON last-trial numbers earlier in this file are superseded and kept
only for comparison. Same binary throughout,
`21a6bd08b37b54d2b8a20d8924a2bbb2`, checked at every cell; no compiler change
and no rebuild.

## Protocol

Per model: warm up ONCE with adaptation on, save the adapted state (step size,
dense inverse mass, post-warmup draw); then measure BOTH arms with adaptation
off from that shared state with matched seeds. This keeps the realistic adapted
regime while removing the trajectory-length divergence that made the old
statistic unsafe. 20 draws, five timed executions of the same compiled thunk
**in one process** with the first call discarded; median and geomean reported.

Plumbing added, all ADDITIVE, no compiler change (the resume API already
existed: trace-based `ProbProg.mcmc` returns a 5-tuple ending in an
`MCMCState`, and `mcmc(state::MCMCState, ...)` already defaults to adaptation
off with the state's step size and inverse mass):

- `harness.jl`: `nuts_state_program` (identical to `nuts_program` but also
  returns the post-run state's step size and inverse mass), `--save-state`,
  `--load-state`, `--repeat N`, and `last_draw_init_params`. `nuts_program` and
  the whole measured path are untouched.
- The injected POSITION reuses the existing `init_params` route that pplbench
  already drives every trial, in constrained space, so no
  unconstrained/constrained conversion is introduced.

Two defects were found and fixed while building this, both of the silent kind:

1. **JSON flattening of the dense inverse mass.** `JSON3.write` serialized the
   58x58 matrix as a flat 3364-element column-major array, which the loader
   would then have read as a length-3364 *diagonal* vector. Caught by the
   round-trip check on the first run. Fixed by writing nested rows plus an
   explicit `inverse_mass_rank` / `inverse_mass_shape`, and by a hard shape gate
   in the loader that errors rather than proceeding on a mismatch.
2. **Cross-process repeats measured warm-up, not steady state.** Running five
   separate processes makes every execution a first call. On Scotland that read
   1.79x where the steady state is 4.6x. Fixed with in-process `--repeat`.

## Verification on THIS roster (required before any migrated number)

Run on two of this roster's own models, prioritizing constrained supports as
directed (`verify_state_injection.py`, results in `evidence/verify_state_*.json`):

| Check | car_spatial_scotland (P=58) | pgls (P=3) |
|---|---|---|
| Dense mass differs from its own diagonal | max off-diag 0.5268 | 0.00996 |
| Exact re-injection reproduces the run | bit-identical (0) | bit-identical (0) |
| Control: identity mass | changes output (froze chain: degenerate) | changes output (froze) |
| Control: diagonal-only mass | changes output (froze: degenerate) | changes output, chain healthy |
| **Control: off-diagonal scaled 0.9 (non-degenerate)** | **changes output by 5.72, chain healthy 20/20** | **changes by 1.05, healthy 20/20** |
| Control: half step size | changes output, chain healthy | changes output, chain healthy |
| Arms from shared state | agree to 2.35e-12 relative | 1.36e-15 relative |

The off-diagonal control is the one that matters. The obvious controls
(identity, diagonal-only) do change the output, but on these models they do it
by freezing the chain, which proves consumption only degenerately and cannot
distinguish dense from diagonal consumption. Scaling only the off-diagonal
keeps `diag(M)` exactly and stays SPD, so the chain stays healthy and a
diagonal-only consumer would have reproduced the baseline exactly. It did not,
so the dense off-diagonal is genuinely consumed at both P=3 and P=58.

## Results (batch frozen1, 17:12-17:53)

| Pattern | Model | Size | SICM ON (s) | SICM OFF (s) | median | geomean |
|---|---|---|---|---|---|---|
| CholScale | Lip Cancer CAR | n=56 | 0.0810 | 0.4093 | **5.05x** | 5.20x |
| CholScale | NYC Traffic CAR | n=250 | 7.803 | 36.256 | **4.65x** | 4.47x |
| TriSolveScale | Shorebird PGLS | n=71 | 0.00173 | 0.02109 | **12.16x** | 11.13x |
| OuterProdScale | Hier. MVN | K=60, J=2 | 0.4998 | 2.2978 | **4.60x** | 4.59x |
| EigenLift | Radon LMM | n=250 | 0.01078 | 0.2796 | **25.94x** | 24.94x |
| DotScale | Columbus SAR (not plotted) | n=49 | 0.00992 | 0.01053 | **1.06x** | 1.07x |
| GaussianSufficientStats | Diamonds Regr. | N=5000, K=25 | 0.00896 | 0.02755 | **3.07x** | 3.00x |

Adapted step sizes are 2.0x to 15.9x larger than the fixed steps the old
protocol used, so trees are shallower and absolute times are much smaller. The
RATIOS are the comparable quantity.

### Comparison with the superseded numbers

| Row | naive (adapt ON, last trial) | 300-draw adapt-off | **frozen state** |
|---|---|---|---|
| Lip Cancer CAR | 5.5x | 5.7x | **5.05x** |
| NYC Traffic CAR | 3.9x | not run | **4.65x** |
| Shorebird PGLS | 18.1x | 10.3x | **12.16x** |
| Hier. MVN | 4.6x | 5.0x | **4.60x** |
| Radon LMM | 43.1x | 34.2x | **25.94x** |
| Columbus SAR | 40.7x | 1.08x | **1.06x** |
| Diamonds Regr. | 10.0x | not run | **3.07x** |

Four rows are stable across protocols. Three move materially and all three move
DOWN: Radon LMM 43 to 26, Diamonds 10 to 3.1, and Columbus SAR 40.7 to 1.06.
The old statistic was inflating them, most extremely the SAR row, whose two arms
have identical LAPACK inventories and whose 40.7x was entirely trajectory
divergence. This is the quantitative vindication of the
wall-and-IR-must-agree rule.

## Gates

- **Contamination.** External-CPU gate adopted from the pre-lowering campaign:
  summed %CPU of processes not owned by us, sampled every 5 s for the duration
  of every cell, cell retried if it ever exceeds 1000%. It fired three times
  (car_spatial_nyc_sicm at 5459%, lmm_radon_sicm at 1409%, diamonds_sicm twice
  at 3758% and 6405%), all retried to a clean pass. Raw loadavg would have
  missed all four: the machine carried a 460-960% external baseline throughout
  from other users' bazel and julia jobs. Per-cell records in
  `logs_frozen/frozen1/batch_meta.txt`.
- **Binary.** md5 recorded per cell, unchanged at all 16 recorded checkpoints.
- **Matched trajectories.** 5 of 7 rows pass at 1e-16 to 3e-12 relative. **Two
  FAIL and are reported as failing**: Hier. MVN (1.5e-5) and Columbus SAR
  (3.1e-1). Cause located per draw: the arms agree to ~1e-12 and then a ULP
  difference flips a discrete sampler decision, at draw 7 for Hier. MVN and at
  draw 0 for Columbus SAR, after which the chains decorrelate.

  This is a limitation of the protocol **for this ablation specifically** and
  is worth carrying forward. The pre-lowering ablation changes only which
  loop-invariant computations are hoisted, preserving the arithmetic, so its
  arms can be bit-identical. SICM changes the arithmetic itself
  (`chol(s*A)` becomes `sqrt(s)*chol(A)`), so its arms differ at the ULP floor
  by construction, and any discrete decision downstream can amplify that into a
  full trajectory divergence. Freezing the adapted state removes the systematic
  divergence but cannot remove this one.

  Neither failure changes a conclusion: both ratios are corroborated by the
  independent 300-draw adaptation-off measurement (Hier. MVN 5.0x against
  4.60x; SAR 1.08x against 1.06x).

## Figure

Six rows, unchanged membership. The DotScale row stays out: its honest number
is now 1.06x on the figure's own protocol, and a 1.06x bar labelled DotScale
would read as "DotScale is worthless" when what it measures is a cascade whose
EigenLift half is blocked by the diagonal-extraction coverage gap. Recorded
here instead. If that gap is fixed the row can be measured again and added.

`figures/gen_sicm_ablation_bar.py` changed only in axis limits, the speedup
column position, and `fmt_s`, which now prints three significant figures: under
this protocol several cells are single-digit milliseconds and the previous
two-decimal formatter would have printed "0.00 s" for them.

## Files added by the migration

- `harness.jl`: `nuts_state_program`, `last_draw_init_params`, `--save-state`,
  `--load-state`, `--repeat` (all additive).
- `ablation/sicm_roster/run_frozen_batch.sh`, `verify_state_injection.py`,
  `summarize_frozen.py`.
- `ablation/sicm_roster/adapted_states/*.json` (7 saved states),
  `data/*.json` (7 harness data files), `logs_frozen/frozen1/`,
  `frozen_state_RAW_2026-08-25.json`,
  `evidence/verify_state_{car_spatial_scotland,pgls}.json`.
