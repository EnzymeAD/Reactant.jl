# Real datasets used by the PPLBench models

Every file in this directory is observed data. No file here is simulated, and no
model that loads one of them simulates its response variable: `generate_data`
returns the recorded outcome vector as `Y`. The `seed` argument of
`generate_data` is therefore unused for these branches, except where a
subsample is drawn (see *Subsample policy* below).

The files are checked in as JSON so that the benchmark has no R, `ape`, `brms`,
`posteriordb`, or network dependency at run time.

---

## `scotland_lip_cancer.json`

**Used by** `models/car_spatial.py`, branch `dataset="scotland"`
(Leroux CAR disease mapping; `sicm/car_spatial.jl` on the Impulse side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `N` | scalar | 56 districts |
| `y` | (56,) | observed lip cancer cases per district, 1975-1980 |
| `E` | (56,) | expected cases from indirect age/sex standardization |
| `x` | (56,) | percent of the workforce in agriculture, fishing and forestry (AFF) |
| `W` | (56, 56) | district adjacency, symmetric 0/1, zero diagonal, 132 undirected edges |

**Provenance.** Scottish lip cancer counts by district, the standard worked
example of areal disease mapping. Introduced by Clayton and Kaldor (1987),
re-analysed by Breslow and Clayton (1993), and distributed for decades as the
`lips` example of the BUGS/WinBUGS example volumes. The copy here was taken
verbatim from the Stan ICAR case study data file
`knitr/car-iar-poisson/scotland_data.R` in `stan-dev/example-models`
(Morris, Wheeler-Martin, Simpson, Mooney, Gelman and DiMaggio, *Bayesian
hierarchical spatial models: implementing the Besag York Mollie model in Stan*,
Spatial and Spatio-temporal Epidemiology, 2019).

References:
- Clayton, D. and Kaldor, J. (1987). Empirical Bayes estimates of age-standardized
  relative risks for use in disease mapping. *Biometrics* 43(3), 671-681.
- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in generalized
  linear mixed models. *JASA* 88(421), 9-25.
- Morris, M. et al. (2019). Bayesian hierarchical spatial models: implementing
  the Besag York Mollie model in Stan. *Spat. Spatio-temporal Epidemiol.* 31.

**What the model does with it.** The benchmark forms the Leroux precision
`Q = rho (D - W) + (1 - rho) I` with `D = diag(rowsum(W))` and `rho = 0.99`
fixed, inverts it, symmetrizes, and adds the same `1e-6` diagonal jitter the
synthetic branch uses; `log_E = log(E)` is the offset and `Y = y`. `Q_inv` is
sample-invariant, which is what makes the model a `CholScale` instance:
`chol(sigma_phi^2 * Q_inv) -> sigma_phi * chol(Q_inv)`.

**`x` is deliberately unused.** The benchmark model is the intercept-plus-
spatial-field specification `y_i ~ Poisson(E_i exp(alpha + phi_i))`, without the
AFF covariate. The column is kept in the JSON so the file stays a faithful copy
of the source data, not so the model reads it.

---

## `nyc_traffic_submap.json`

**Used by** `models/car_spatial.py`, branch `dataset="nyc_traffic"`
(Leroux CAR disease mapping at a larger `n`; `sicm/car_spatial.jl` on the
Impulse side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `N` | scalar | 250 census tracts (a sub-map; see below) |
| `y` | (250,) | observed traffic-accident counts per tract |
| `E` | (250,) | exposure per tract (the model's Poisson offset) |
| `W` | (250, 250) | tract adjacency, symmetric 0/1, zero diagonal, 630 undirected edges |
| `tract_index` | (250,) | 0-based indices of the selected tracts in the source file |
| `_provenance` | string | source, sub-map rule and sizes, restated in the file itself |

**Provenance.** The posteriordb data set `traffic_accident_nyc`, which backs the
posterior `traffic_accident_nyc-bym2_offset_only`, read from
`posterior_database/data/data/traffic_accident_nyc.json.zip`. The source file
carries `N = 1921` census tracts, `N_edges = 5461` undirected adjacency edges as
1-based `node1`/`node2` index lists, observed counts `y`, exposure `E`, and the
BYM2 `scaling_factor` (unused here).

**Sub-map rule (deterministic; see *Subsample policy*).** Breadth-first search
from tract 0 (0-based; node id 1 in the file's 1-based edge lists), expanding
each frontier node's neighbours in ascending node-id order, until 250 distinct
tracts are collected; the collected ids are then sorted ascending, and `y`, `E`
and the adjacency are restricted to the induced subgraph. The result is one
contiguous piece of the city map: 250 tracts, 630 induced undirected edges,
degrees 1 to 12 (median 5), no isolated tract. `y` and `E` are the observed
values, unmodified; nothing is simulated. Built by
`benchmark/probprog/ablation/sicm_roster/build_nyc_submap.py`.

**Model vs. data: this is not posteriordb's model.** posteriordb's reference
implementation for these data is **BYM2** (a scaled ICAR component plus an
independent component, whose spatial density is written as a sparse sum of
pairwise squared differences with the ICAR scaling factor above). BYM2 involves
no matrix factorization at all, which is exactly why it is not the model here.
The benchmark instead fits the **Leroux CAR** model
(Leroux, Lei and Breslow 2000), a standard proper alternative in areal disease
mapping, on the same data: `phi ~ MVN(0, sigma_phi^2 Q^-1)` with
`Q = rho (D - W) + (1 - rho) I` and `rho = 0.99` fixed. That is the
specification with a dense `n x n` covariance, and hence with the
per-iteration Cholesky that the `CholScale` ablation is about. The data are
posteriordb's; the model is not. Any comparison against a posteriordb reference
posterior would be a comparison of two different models and is not made here.

**References**
- Leroux, B. G., Lei, X. and Breslow, N. (2000). Estimation of disease rates in
  small areas: a new mixed model for spatial dependence. In *Statistical Models
  in Epidemiology, the Environment, and Clinical Trials*, 179-191.
- Riebler, A., Sorbye, S. H., Simpson, D. and Rue, H. (2016). An intuitive
  Bayesian spatial model for disease mapping that accounts for scaling.
  *Statistical Methods in Medical Research* 25(4), 1145-1165. (BYM2)
- Magnusson, M., Burkner, P.-C., Vehtari, A. et al. posteriordb, data set
  `traffic_accident_nyc`.

---

## `shorebird_pgls.json`

**Used by** `models/pgls.py` (phylogenetic generalized least squares;
`sicm/pgls.jl` on the Impulse side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `N`, `K` | scalars | 71 species, 2 columns of X |
| `species` | (71,) | tip labels, in the order the rows use |
| `y` | (71,) | log(Egg.Mass), the response |
| `X` | (71, 2) | [1, log(M.Mass)] |
| `C` | (71, 71) | Brownian-motion phylogenetic correlation of the tree |
| `log_F_Mass` | (71,) | log(F.Mass), carried but not used by the fitted model |
| `gls_beta_reference`, `gls_se_reference`, `ols_beta_reference` | (2,) | reference fits, for the sanity gate |

**Provenance.** The `shorebird` data of the R package **caper** (Orme et al.),
`data/shorebird.rda` of github.com/davidorme/caper: `shorebird.tree`, an ape
`phylo` with 71 tips and 62 internal nodes, and `shorebird.data`, 71 species by
(Species, M.Mass, F.Mass, Egg.Mass, Cl.size, Mat.syst). The traits are from
Lislevand and Thomas, *Limited male incubation ability and the evolution of egg
size in shorebirds* (Biology Letters, 2006/2007).

**Model mirrored.** The caper vignette's simple PGLS,
`pgls(log(Egg.Mass) ~ log(M.Mass), shorebird)` (`vignettes/caper.rnw:530`), at
the default `lambda = 1`, i.e. Brownian motion. At lambda = 1 the correlation
structure is the plain phylogenetic variance-covariance of the tree, so `C` is
fixed data. (The vignette also fits a `lambda='ML'` variant and a
`log(M.Mass) * log(F.Mass)` variant; neither is used here, because estimating
lambda would make C sample-dependent and remove the invariant the row is about.
`log_F_Mass` is shipped so that variant remains reproducible.)

**How C is built.** ape's `(edge, edge.length)` representation is converted to
Newick and run through the Newick parser of `models/phylo_pois.py`, so both
phylogenetic rows in this benchmark build their covariance with identical code.
`C[i, j]` is the root-to-MRCA path length. The result is cross-checked against
an independent direct edge-matrix computation (agreement: exact, max abs
difference 0). The tree is ultrametric (root-to-tip 81.872 for every tip), so
after dividing by that depth `diag(C) = 1` and `C` is a correlation matrix;
`cond(C) = 1099`, Cholesky succeeds. Nothing is simulated.

**Reference values** (what the fitted posterior should reproduce): PGLS
`beta = [-0.4218, 0.6756]` with standard errors `[0.2278, 0.0348]`; OLS
`beta = [-0.3287, 0.6795]`. The slope near 0.68 is the classic allometric
scaling exponent of egg mass on body mass.

**Note on the density's spelling.** The benchmark writes the GLS density in
whitened-Cholesky form (whiten y and X separately against `chol(sigma^2 C)`).
That is one of the two standard GLS spellings and the numerically preferred
one, used by Stan's `multi_normal_cholesky` and by brms's phylogenetic models.
It is **not** what caper's own `pgls` does: caper forms the explicit inverse
`iV <- solve(V)` and accumulates `X' iV X` and `X' iV y`
(`caper/R/pgls.R:383-393`), using a whitening matrix only to compute
phylogenetic residuals. The model being fitted is the vignette's; the spelling
of its density is this benchmark's choice.

---

## `boston_sar.json`

**Used by** `models/sar_lag.py` (Bayesian spatial-lag / SAR model;
`sicm/sar_lag.jl` on the Impulse side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `N`, `K` | scalars | 506 census tracts, 14 columns of X |
| `y` | (506,) | log(CMEDV), the response |
| `X` | (506, 14) | intercept plus 13 z-scored hedonic regressors |
| `feature_names` | (14,) | column names |
| `W` | (506, 506) | row-standardised contiguity, the model's lag operator |
| `Ws` | (506, 506) | symmetric normalised adjacency, used for the log-determinant |
| `rho_lo`, `rho_hi` | scalars | the stationary interval for rho |
| `profile_rho_hat`, `profile_sigma_hat`, `profile_beta_hat` | | reference fit, for the sanity gate |

**Provenance.** The `boston` data of the R package **spData**
(github.com/Nowosad/spData, `data/boston.rda`): `boston.c`, the 506-tract
Harrison and Rubinfeld (1978) housing table carrying Gilley and Pace's
corrected median value CMEDV, and `boston.soi`, the sphere-of-influence
neighbours list for those tracts (an spdep `nb` object, already symmetric,
1076 undirected edges, degrees 1 to 8, no isolated tract). The regressors are
the Harrison and Rubinfeld hedonic set used in spdep's own Boston examples:
CRIM, ZN, INDUS, CHAS, NOX^2, RM^2, AGE, log(DIS), log(RAD), TAX, PTRATIO, B,
log(LSTAT). Non-intercept columns are z-scored, which rescales beta and helps
the sampler but does not change the model. `y` and `X` are recorded values;
nothing is simulated.

**Two weight matrices, and why.** The model equation is
`(I - rho W) y = X beta + eps`, and its likelihood needs `log|I - rho W|`.

- `W = D^-1 B` is the **row-standardised** contiguity. It is the lag operator
  of the model equation and the convention the Boston SAR literature reports
  rho in. It is not symmetric.
- `Ws = D^-1/2 B D^-1/2` is the **symmetric normalised adjacency**. It is
  similar to W (`Ws = D^1/2 W D^-1/2`), hence has exactly the same spectrum,
  hence `log|I - rho W| = log|I - rho Ws|` for every rho. The model takes the
  log-determinant as a Cholesky of `I - rho Ws`, which is symmetric positive
  definite on the stationary interval.

This is the standard computational device for the SAR determinant (Ord 1975;
spdep's `similar.listw`), not a modelling shortcut: it is used precisely
because the symmetric form makes the determinant tractable. The identity was
verified numerically to 1.7e-13 over the stationary interval when the file was
built. Two alternatives were tried and rejected: a plain symmetric binary W and
a spectrally normalised symmetric binary W both give a degree-weighted, and so
badly behaved, lag operator, driving the profile-likelihood rho to 0.019-0.096
against the literature value, because they are not row-standardised.

`eig(Ws) = [-0.970864, 1.000000]`, so the stationary interval is
`rho in (-1.0300, 1.0000)`; `I - rho Ws` is positive definite and factorizes
throughout it (condition number 1 at rho = 0 rising to 196 at rho = 0.99).

**Reference values** (what the fitted posterior should reproduce): profile
likelihood `rho_hat = 0.4856`, `sigma_hat = 0.1388`, matching the rho of about
0.45-0.50 that spdep's Boston `lagsarlm` examples report; and the classic
hedonic signs, NOX^2 negative, RM^2 positive, log(LSTAT) negative, CRIM
negative. Moran's I of the OLS residuals is 0.41 (binary W) / 0.45
(row-standardised W) against an expectation of -0.002, so the spatial
dependence the model is fitting is real and strong.

---

## `radon_mn.json`

**Used by** `models/linear_mixed_model.py`, branch `dataset="radon_mn"`
(marginalized county-intercept LMM; `sicm/linear_mixed_model.jl` on the Impulse
side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `N` | scalar | 919 homes |
| `J` | scalar | 85 Minnesota counties |
| `log_radon` | (919,) | log radon measurement (pCi/L), the response |
| `floor_measure` | (919,) | 0 = basement, 1 = first floor |
| `log_uppm` | (919,) | log county-level soil uranium, a county-level predictor |
| `county_idx` | (919,) | 1-based county index |

**Provenance.** The Minnesota subset of the EPA National Residential Radon
Survey, the running example of Gelman and Hill (2007) chapters 12-13 and the
canonical multilevel-model test case. The field names and the 919/85 shape are
those of the `radon_mn` posteriordb dataset.

References:
- Gelman, A. and Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press. (radon, ch. 12-13)
- Magnusson, M., Burkner, P.-C., Vehtari, A. et al. posteriordb: a database of
  Bayesian posterior inference, `radon_mn` data set.

**What the model does with it.** `X = [1, floor_measure]`, `Z` = county
indicator matrix, `K_re = Z Z'`, `Y = log_radon`, and the county intercepts are
marginalized out, giving `y ~ MVN(X beta, sigma_u^2 K_re + sigma_e^2 I)`. `K_re`
is sample-invariant, which is what makes the model an `EigenLift` instance:
`chol(s K_re + t I)` is answered from a hoisted eigendecomposition of `K_re`.
`log_uppm` is not used by this specification (no county-level predictor).

---

## `diamonds.json`

**Used by** `models/diamonds_regression.py`
(brms log-log price regression; `sicm/diamonds_regression.jl` on the Impulse
side).

**Contents**

| key | shape | meaning |
|---|---|---|
| `_provenance` | string | source and sizes, restated in the file itself |
| `N` | scalar | 5000 diamonds |
| `K` | scalar | 25 population-level effects, including the intercept |
| `Y` | (5000,) | log price, the response |
| `X` | (5000, 25) | brms population-level design matrix; column 0 is the intercept column of ones, columns 1 to 24 are log(carat), the spline and polynomial terms in the 4C grades, and the physical dimensions |
| `prior_only` | scalar | 0, carried through from the source file (fit the likelihood) |

**Provenance.** The `diamonds` example of the ggplot2 data set as fitted by brms,
distributed as the posteriordb posterior `diamonds-diamonds`. Copied verbatim
from `posterior_database/data/data/diamonds.json.zip`; the accompanying model is
`posterior_database/models/stan/diamonds.stan`, generated with brms 2.10.0.
`Y` and `X` are the recorded regression variables, unmodified; nothing is
simulated.

References:
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*.
  Springer. (the `diamonds` data set)
- Buerkner, P.-C. (2017). brms: An R package for Bayesian multilevel models
  using Stan. *Journal of Statistical Software* 80(1), 1-28.
- Magnusson, M., Buerkner, P.-C., Vehtari, A. et al. posteriordb: a database of
  Bayesian posterior inference, posterior `diamonds-diamonds`.

**What the model does with it.** The Stan file centers every non-intercept
column of `X` at its own column mean (its `transformed data` block) and splits
the intercept out as a separate parameter, so its mean is
`Intercept + Xc b`. The benchmark folds the intercept back into the design
matrix: with `X_aug = [1, Xc]` and `beta = [Intercept; b]` the mean is the
single product `X_aug beta`, which is the identical likelihood with the
identical parameters and the identical priors, written the way a statistician
writes a regression down rather than the way brms code-generates one. `X_aug`
and `Y` are sample-invariant, which is what makes the model a
`GaussianSufficientStats` instance: the residual sum of squares
`||Y - X_aug beta||^2` is answered from a hoisted thin QR of `X_aug`, dropping
the per-gradient cost from `O(N K)` to `O(K^2)`.

**One prior substitution.** `diamonds.stan` puts a half student-t(3, 0, 10) on
`sigma`. ProbProg has no half student-t, so the benchmark uses `HalfCauchy(10)`,
the half student-t with one degree of freedom instead of three, at the same
scale and on the same support. Both are weakly informative on a residual SD
that 5000 observations at `R^2 = 0.986` pin tightly, so the choice does not move
the posterior. The `Intercept ~ student_t(3, 8, 10)` and `b ~ normal(0, 1)`
priors are carried over exactly.

---

## `coal_mining_jarrett.json`

**Used by** `models/coal_mining.py`, branch `dataset="jarrett"` (change point
with the discrete switch marginalized, NUTS on the two rates;
`standard/coal_mining.jl` on the Impulse side, `ppls/numpyro/coal_mining.py`
on the NumPyro side), and by the programmable MH harnesses of the generality
figure under `COAL_DATA=jarrett` (`impulse_coal_mh_masked.jl`,
`gen_coal_mh_vec_harness.jl`, `gen_coal_mh_harness.jl`,
`genjax_coal_mh_harness.py`).

**Contents**

| key | shape | meaning |
|---|---|---|
| `years` | (111,) | calendar years 1851 to 1961 |
| `counts` | (111,) | recorded coal-mining disasters per year; `null` for the two missing years |
| `missing_years` | (2,) | 1890 and 1934 |
| `first_year`, `last_year`, `n_years` | scalar | 1851, 1961, 111 |
| `_provenance` | string | source and missing-year handling, restated in the file itself |

**Provenance.** Yearly counts of British coal-mining disasters, transcribed
from the PyMC listing as printed in Salvatier, Wiecki and Fonnesbeck (2016,
PyMC3 case study 2, p15 to 16, a NumPy masked array with `-999` as the missing
sentinel) and checked value by value against the same listing in Abril-Pla et
al. (2023, Code 1, p13 to 14, a pandas Series with `np.nan`). The two listings
agree on all 111 entries; 109 years are observed and sum to 188 disasters.
Both papers cite Jarrett (1979) as the source, which corrected the widely
reprinted series of Maguire, Pearson and Wynn (1952). The values here were not
re-transcribed from Jarrett's own table.

References:
- Jarrett, R. G. (1979). A note on the intervals between coal-mining disasters.
  *Biometrika* 66(1), 191-193.
- Maguire, B. A., Pearson, E. S. and Wynn, A. H. A. (1952). The time intervals
  between industrial accidents. *Biometrika* 39(1/2), 168-180.
- Salvatier, J., Wiecki, T. V. and Fonnesbeck, C. (2016). Probabilistic
  programming in Python using PyMC3. *PeerJ Computer Science* 2:e55.
- Abril-Pla, O. et al. (2023). PyMC: a modern and comprehensive probabilistic
  programming framework in Python. *PeerJ Computer Science* 9:e1516.

**Missing years.** A missing year contributes no likelihood term. Every other
year keeps its calendar position `t = year - 1851`, and the switch ranges over
all 112 boundaries `s = 0..111` (the late regime starts in year `1851 + s`).
This is the posterior over the switch and the two rates that the PyMC papers'
imputation gives, because an unobserved Poisson count sums to one; it is also
the one handling that all four implementations can share without an extra
latent per missing year. The MH harnesses pass only the 109 observed counts
together with their year indices (the rate for year `t` is the early rate when
`t < s`). The PPLBench model keeps all 111 positions, fills the two missing
counts with 0 and carries a 0/1 variable `observed`; the marginalized
likelihood and the predictive log-likelihood multiply every per-year term by
it (Impulse folds the mask into its constant prefix and suffix matrices).

**What the model does with it.** The model is the existing benchmark
specification, unchanged: `l1, l2 ~ Gamma(3, 1)`, `s ~ DiscreteUniform{0..T}`
with `T = 111`, `y_t ~ Poisson(t < s ? l1 : l2)`. The priors are the
benchmark's, not the PyMC papers' `Exponential(1)`.

---

## Subsample policy

Two of the benchmark rows need an `n` small enough that the SICM-off arm, which
redoes an `O(n^3)` factorization at every leapfrog step, finishes in a
reasonable wall clock, while staying large enough for the factorization to
dominate. Scotland and diamonds are used whole; the NYC and radon rows are
reduced, by two different rules, because the two data sets have different
structure.

- **Scotland (n = 56): no subsampling.** All 56 districts, all 132 edges, the
  complete observed count vector.
- **Diamonds (N = 5000): no subsampling.** All 5000 rows and all 25 columns of
  the brms design matrix. This row's SICM-off arm costs `O(N K)` per gradient,
  not `O(N^3)`, so the full data set is affordable in both arms, and the whole
  point of the row is that `N` is large relative to `K`.
- **NYC traffic (n = 250 of 1921): deterministic contiguous sub-map, no
  randomness.** A CAR model's covariance is defined by a *map*, so a random
  subset of tracts would shatter the adjacency graph and change the model's
  character. The reduction is therefore a breadth-first search from tract 0
  over the recorded edge list, neighbours expanded in ascending node-id order,
  stopped at 250 tracts, keeping the induced subgraph. This is a pure function
  of the source file (no seed at all): every arm sees the same sub-map and a
  re-run reproduces it exactly. The resulting piece is connected, has no
  isolated tract, and keeps a realistic degree profile (1 to 12, median 5)
  against the full map's edge density.
- **Shorebird (n = 71): no subsampling.** All 71 species, the whole tree.
- **Boston (n = 506): no subsampling.** All 506 tracts, all 1076 edges.
- **Radon (n = 250 of 919): fixed-seed subsample.** `numpy.random.default_rng(seed)`
  with the config's model `seed` (12345 for the ablation config) draws
  `n` row indices without replacement; the indices are sorted, so the subsample
  keeps the file's row order. Counties are then re-indexed to the ones present
  in the subsample (`q = 74` at `n = 250, seed = 12345`), so `Z` has no all-zero
  column and `attrs["q"]` reports the count actually used. Nothing else is
  filtered: no county is dropped for being small, and homes are not stratified
  by county or by floor. Because the draw is a pure function of `seed` and `n`,
  every arm of a config sees the identical subsample, and a re-run reproduces it
  exactly.

Both branches keep the same `xarray.Dataset` schema as their synthetic
counterparts (same variable names, dims and `attrs` keys), so every downstream
consumer (the Impulse/NumPyro/Stan wrappers, the PLL evaluator, the init-param
generator) is unchanged.
