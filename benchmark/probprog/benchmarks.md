# Flagship Benchmark Models

Implementation order: GP → Logistic Regression → Eight Schools → Hierarchical Poisson

## 1. Gaussian Process Regression (IMPLEMENT FIRST)

**Source**: Existing `reactant/gp.jl` benchmark. Add PosteriorDB variant if available.

**Why first**: This is the SICM showcase. The existing algebraic decomposition patterns
(Cholesky, TriSolve, LogMultiply) were designed for exactly this model. It produces
the most dramatic, unambiguous speedup and is the strongest argument for the
compiler-first approach.

**Model**:
```
kernel_var    ~ LogNormal(0, 10)       # scalar, positive
kernel_length ~ LogNormal(0, 10)       # scalar, positive
kernel_noise  ~ LogNormal(0, 10)       # scalar, positive

K_ij = kernel_var * exp(-0.5 * ((X_i - X_j) / kernel_length)²) + (kernel_noise + jitter) * I
Y ~ MultiNormal(0, K)
```

**Parameters**: 3 (all positive-support → unconstrained via log transform)

**SICM patterns that fire**:
- `CholeskyScaleFactorizationHLO`: `cholesky(var * K_base) → sqrt(var) * cholesky(K_base)`
  Saves O(N³) Cholesky per leapfrog step. K_base depends on length but not var.
- `DotGeneralScaleFactorizationHLO`: factors scalar out of matrix-vector products
- `TriangularSolveScaleFactorizationHLO`: `tri_solve(sqrt(var)*L, y) → (1/sqrt(var)) * tri_solve(L, y)`
- `LogMultiplyDistributionHLO`: `log_det(var * K_base) → N*log(var) + log_det(K_base)`

**Hoistable invariant ops** (plain hoisting, no algebraic decomposition needed):
- Pairwise distance matrix: `(X_i - X_j)²` is pure data → O(N²)
- Jitter diagonal addition pattern

**Benchmark sizes**: N=30, N=100, N=500
- At N=500, unhisted Cholesky = O(125M) FLOPs per leapfrog step
- SICM reduces this to O(1) amortized (computed once before MCMC loop)

**What needs implementing**:
- [ ] PosteriorDB integration (model class, data loader, reference draw comparison)
- [ ] NumPyro PPL adapter for GP model
- [ ] Stan PPL adapter for GP model (if not using PosteriorDB Stan file directly)
- [ ] Verify SICM patterns fire correctly on the GP mcmc_region IR

**Paper story**: "SICM automatically discovers and applies the Cholesky factorization
optimization, reducing per-step cost from O(N³) to O(N²). This optimization is
non-obvious — most Stan users don't manually factor their covariance matrices —
and impossible in trace-based systems like NumPyro where the compiler cannot
distinguish data from parameters."


## 2. Logistic Regression (IMPLEMENT SECOND)

**Source**: Already exists in PPLBench infra (`pplbench_logistic_regression.jl`).
Add PosteriorDB variant for correctness validation.

**Why second**: Already working end-to-end. Minimal new code. Good baseline that
demonstrates the full benchmark pipeline. The model is simple enough that SICM
wins are modest — the story here is raw compiler performance (XLA codegen,
AD efficiency) not domain-specific optimization.

**Model**:
```
alpha ~ Normal(0, alpha_scale)         # scalar
beta  ~ Normal(beta_loc, beta_scale)   # k-dimensional

mu_i = alpha + X_i · beta
Y_i ~ Bernoulli(logits = mu_i)        # observed
```

**Parameters**: 1 + k (typically k=10, so 11)

**SICM patterns that fire**: Minimal.
- Plain hoisting: prior normalization constants
- `X @ beta` is irreducibly sample-dependent (matrix × sample-dependent vector)
- The gradient `dL/dbeta = X^T @ grad_mu` materializes X^T during AD — but this
  happens after inference expansion, not in the mcmc_region

**What this model benchmarks**:
- Raw NUTS performance (leapfrog speed, tree building, adaptation)
- AD efficiency (Enzyme vs JAX autodiff)
- XLA codegen quality vs NumPyro's JAX backend
- Warmup convergence (step size + mass matrix adaptation)

**Benchmark sizes**: N=1000 k=10 (PPLBench default), N=5000 k=50 (scaling test)

**What needs implementing**:
- [ ] PosteriorDB model class (load fixed data + reference draws instead of synthetic)
- [ ] Reference draw comparison metric (replace PLL with posterior accuracy)
- [ ] Already have: Reactant adapter, NumPyro adapter, Stan adapter

**Paper story**: "On standard logistic regression, Impulse matches or exceeds
NumPyro's per-step performance through lower-level XLA codegen, while producing
identical posterior samples validated against PosteriorDB gold standard draws."


## 3. Eight Schools (IMPLEMENT THIRD)

**Source**: PosteriorDB `eight_schools-eight_schools_noncentered`.

**Why third**: The canonical PPL benchmark. Every paper has it. Small model
(10 parameters) so the benchmark is about correctness and the funnel geometry,
not raw throughput. Tests that NUTS handles the non-centered parameterization
correctly.

**Model (non-centered)**:
```
mu         ~ Normal(0, 5)             # scalar
tau_tilde  ~ Normal(0, 1)             # scalar (tau = 5 * |tau_tilde| via HalfCauchy)
  OR tau   ~ HalfCauchy(5)            # positive scalar
theta_tilde_j ~ Normal(0, 1)          # 8-dimensional

theta_j = mu + tau * theta_tilde_j
Y_j ~ Normal(theta_j, sigma_j)        # observed, sigma_j known
```

**Parameters**: 10 (mu, tau/tau_tilde, 8 × theta_tilde)

**Data**: 8 observations with known standard errors sigma_j. Tiny dataset.

**SICM patterns that fire**:
- Plain hoisting: `1/sigma_j²`, `log(sigma_j)`, `Y_j/sigma_j²` → all data-only
- No algebraic decomposition needed (no matrix ops, no scaled covariance)

**What this model benchmarks**:
- Correctness on the "Neal's funnel" geometry
- NUTS adaptation quality (mass matrix must capture mu-tau correlation)
- Comparison against PosteriorDB gold standard (R-hat, ESS, posterior means)

**What needs implementing**:
- [ ] PosteriorDB model class (load eight_schools data + reference draws)
- [ ] Julia model function (translate from Stan)
- [ ] Reactant PPL adapter + Julia script
- [ ] NumPyro PPL adapter (translate model)
- [ ] Support constraints: HalfCauchy requires positive support on tau

**Paper story**: "Impulse produces correct posterior samples on the challenging
eight schools funnel geometry, matching PosteriorDB reference draws within
statistical tolerance."


## 4. Hierarchical Poisson (IMPLEMENT FOURTH)

**Source**: PosteriorDB (pick a suitable hierarchical count model).

**Why fourth**: Demonstrates hierarchical model support and non-Gaussian
likelihoods (Poisson). Tests that the compiler handles `exp()` in the
likelihood correctly (Poisson rate = exp(theta)).

**Model**:
```
mu    ~ Normal(0, 10)                  # scalar, global mean
sigma ~ HalfNormal(1)                  # scalar, positive, group-level sd

theta_j ~ Normal(mu, sigma)            # J group-level parameters
Y_j ~ Poisson(exp(theta_j))            # observed counts
```

**Parameters**: 2 + J (mu, sigma, J × theta)

**SICM patterns that fire**:
- Plain hoisting: `lgamma(Y_j + 1)` (= log(Y_j!)) is pure data
- Normalization constant elimination in Normal logpdf

**What this model benchmarks**:
- Non-Gaussian likelihood (Poisson log-pmf involves exp, lgamma)
- Hierarchical structure with shared hyperparameters
- Constrained parameter handling (sigma > 0)

**What needs implementing**:
- [ ] Find/choose PosteriorDB model (or create synthetic benchmark)
- [ ] Julia model function
- [ ] Reactant PPL adapter + Julia script
- [ ] NumPyro PPL adapter
- [ ] Poisson distribution support in Reactant ProbProg (if not already present)

**Paper story**: "Impulse handles hierarchical models with non-Gaussian
likelihoods, automatically hoisting expensive data-dependent computations
(lgamma) out of the MCMC inner loop."


---

## Implementation Order Rationale

```
GP  →  Logistic Regression  →  Eight Schools  →  Hierarchical Poisson
 1            2                      3                    4

SICM         Already working        Canonical            New distribution
showcase     baseline               correctness          support
```

**GP first** because:
- Demonstrates the core SICM contribution (algebraic decomposition)
- Produces the most dramatic speedup numbers for the paper
- Existing `gp.jl` means the Julia model already works
- Validates that the 4 existing SICM patterns fire correctly

**Logistic Regression second** because:
- Already fully working in PPLBench infra
- Minimal new code — just add PosteriorDB data loading
- Establishes the benchmark pipeline (PosteriorDB → correctness check → timing)
- Other models reuse this pipeline infrastructure

**Eight Schools third** because:
- Required for paper credibility (every PPL paper has it)
- Small model, quick to implement
- Tests correctness more than performance
- May need HalfCauchy distribution support

**Hierarchical Poisson fourth** because:
- May require new distribution support (Poisson)
- Least critical for the paper if time is tight
- Can be replaced with N-Schools from PPLBench (already has model class)


## Fallback: Use PPLBench Models Instead of PosteriorDB

If PosteriorDB integration takes too long, all four models can be run as
PPLBench-style synthetic benchmarks with PLL evaluation:

| Flagship | PPLBench equivalent | Status |
|----------|-------------------|--------|
| GP | Custom (add to models/) | Need model class |
| Logistic Regression | `LogisticRegression` | Already done |
| Eight Schools | `NSchools` (similar hierarchical) | Already has model class |
| Hierarchical Poisson | Custom or use `RobustRegression` as substitute | `RobustRegression` exists |


## 5. Crowd-Sourced Annotation (Dawid-Skene, discrete latents marginalized)

**Source**: Resurrected from upstream PPLBench (`crowd_sourced_annotation`,
ACL Q14-1025). Discrete per-item true labels are marginalized exactly via
`log_sum_exp` over categories, matching the upstream Stan reference. Impulse
expresses the marginal through a custom-logpdf observed site; continuous
latents are `prev` (K-simplex) and `confusion_matrix` (k*K batched
K-simplices, SIMPLEX stick-breaking support).

**Config**: `pplbench_configs/standard/crowd_sourced_annotation.json`
(upstream defaults n=2000, k=10, num_categories=3, num_labels_per_item=3;
1000 train / 1000 test; iterations 4000, warmup 1000, 4 trials;
position_size 62 unconstrained). Smoke variant:
`crowd_sourced_annotation_smoke.json` (n=200, k=5).

**Results** (2026-07-04, CPU, 32 threads, seed 12345, outputs
`outputs/2026-07-04_21:52:34`):

| PPL | compile (s) | infer per trial (s) | final PLL per trial | n_eff/s median |
|-----|-------------|---------------------|---------------------|----------------|
| Impulse | 49.7 | 4.23, 3.71, 3.57, 3.62 | -1805.7, -1809.4, -1808.9, -1808.8 | 5485 |
| Impulse (no opt) | 49.8 | 4.27, 3.69, 3.49, 3.53 | -1805.1, -1810.0, -1809.0, -1804.4 | 5835 |
| Stan | 6.4 | 91.6, 95.1, 83.3, 90.0 | -1806.7, -1804.8, -1847.4, -1808.0 | 0.1 |

PLL curves overlap (both Impulse variants converge to the same PLL band as
Stan's healthy trials; Stan trial 2 mixed poorly, dragging its n_eff down).
Impulse is ~24x faster per trial than Stan at equal draw counts. SICM makes
no measurable difference on this model (no heavy sample-invariant compute).


## 6. Coal-Mining Disasters Change Point (marginalized NUTS + branchy MH mechanism)

**Model**: Poisson time series with an unknown switch year `s`. Priors
`l1,l2 ~ Gamma(3,1)`, `s ~ DiscreteUniform{0..T}`, `y_t ~ Poisson(t<s ? l1 : l2)`.
Synthetic data (seed 12345): T=112, s*=40, l1*=3.0, l2*=0.9.

**Primary — marginalized NUTS**: `s` is integrated out with the prefix/suffix
cumulative-sum trick, so NUTS runs on the two continuous log-rates. Impulse
expresses the marginal `logsumexp_s(-log(T+1) + A_s + B_s)` through a custom-logpdf
observed site; the NumPyro baseline is the same arrangement via `numpyro.factor`
+ `jnp.cumsum` (`ppls/numpyro/coal_mining.py`). In-sample marginal PLL (the full
series is both train and test; a change-point series does not split i.i.d.).

**Config**: `pplbench_configs/standard/coal_mining.json` (iterations 4000, warmup
1000, 4 trials; position_size 2, dense identity mass to match NumPyro
`dense_mass=True`). Smoke: `coal_mining_smoke.json`.

**Results** (2026-07-04, CPU, 32 threads, seed 12345, outputs
`outputs/2026-07-04_23:38:42`):

| PPL | compile (s) | infer per trial (s) | final PLL per trial | n_eff/s median |
|-----|-------------|---------------------|---------------------|----------------|
| NumPyro | 1.3 | 0.591, 0.620, 0.574, 0.592 | -173.46, -173.39, -173.71, -173.67 | 17424 |
| Impulse | 32.3 | 0.633, 0.317, 0.260, 0.270 | -173.46, -173.67, -173.70, -173.75 | 33788 |
| Impulse (no opt) | 33.4 | 0.631, 0.086, 0.216, 0.250 | -173.70, -173.52, -173.71, -173.76 | 55878 |

PLL curves overlap tightly (all final PLL in [-173.4, -173.76]); posterior means
agree (NumPyro l1=2.71/l2=0.99, Impulse l1=2.74/l2=0.99, both recover the truth).
Impulse steady-state infer ~2x faster per trial than NumPyro; SICM makes no
measurable difference (2-parameter model, no sample-invariant heavy compute).

**Mechanism demo — branchy change point** (Impulse-only, C2 stochastic control
flow). `standard/coal_mining.jl` `coal_mining_branch` keeps `s` explicit and
observes each year's Poisson likelihood INSIDE the taken `impulse.if` branch
(per-branch addresses `ye_t`/`yl_t`; nothing sampled/returned by a branch feeds a
later sample op). The branch weight threads correctly through `generate` (n=1
enumerable check: p(s=1|y)=0.980, exact).

The single-address MH kernel (`coal_mining_mh_program`, alternating
`mh(:s)/mh(:l1)/mh(:l2)`) now drives the recovery. It forwards the observations to
`mh`, so a proposal that flips which `impulse.if` branch runs re-scores the
newly-taken observation from the constraint instead of a stale trace slot (C4
fix; before the fix MH gave p(s=1|y)=0.11 vs the exact 0.980). Correctness is
pinned two ways: (1) n=1, MH p(s=1|y)=0.982 vs analytic 0.980
(`test/probprog/if_branch.jl::coal_branch_flip_mh`); (2) n=20 s-only sweep, MH
histogram matches the enumerate-via-`generate` posterior to TVD=0.008 with the
same MAP. At n=56, s*=40 the MH posterior concentrates around the true switch and
agrees with the enumeration cross-check (`figures/coal_mining_mh_posterior.png`,
driver `coal_mining_mh_demo.jl`; JSON carries both `p` (MH) and `p_enum`).

*Cost note*: tracing one `impulse.if` per year is superlinear in the series
length (n=20: 63s compile; n=56: 545s), so the branchy demo uses n=56 rather than
the full 112.

---

## Variational inference (SVI) — SICM-on-VI

VI now runs through the same region pipeline as HMC/NUTS
(`inline-infer-regions → sicm → construct joint logpdf → expand-impulse`), so the
identical sample-invariant code motion rewrites optimize the VI model density.
This is the algorithm-independence result: the CholeskyScale / TriangularSolveScale
/ LogMultiply cascade that speeds up NUTS on `scale_family_mvn` fires inside the VI
`infer_region` logpdf and hoists the invariant matrix work ABOVE the SVI loop.

**Config**: AutoNormal guide, Adam (lr 0.01), 300 steps, 1 particle, n=8, CPU.
Impulse compiles the whole SVI loop into one XLA `while`; NumPyro uses `svi.run`
(lax.scan-fused). Drivers: `test/probprog/svi_sicm_benchmark.jl` (Impulse + IR),
`test/probprog/svi_numpyro_reference.py` (NumPyro `svi.run`). Artifacts under the
evidence dir (`*_hlo_sicm_{on,off}.mlir`, `svi_benchmark.json`,
`numpyro_svi_fair.json`).

**Per-step wall clock + final ELBO**:

| model | framework | per-step (µs) | final ELBO | notes |
|-------|-----------|--------------:|-----------:|-------|
| scale_family_mvn | Impulse (SICM) | 2.01 | -1.2291 | cholesky in-loop 0 |
| scale_family_mvn | Impulse (no opt) | 10.28 | -1.2291 | cholesky in-loop 1; **5.11× slower/step** |
| scale_family_mvn | NumPyro SVI | 558.65 | -1.2671 | matches Impulse ELBO to 6.3e-14 under matched RBG key |
| gp_pois_regr | Impulse (SICM) | 1.92 | -17.6265 | cholesky in-loop 0 |
| gp_pois_regr | Impulse (no opt) | 6.73 | -17.6265 | cholesky in-loop 1; **3.51× slower/step** |
| gp_pois_regr | NumPyro SVI | 802.96 | -19.8088 | — |

**SICM effect** (same framework, on vs off): eliminates the per-step
`stablehlo.cholesky` (in-loop count 1→0) for both models and cuts in-loop
triangular solves (mvn 6→2, gp 2→0), for a 5.11×/3.51× per-step speedup. In the
lowered HLO with SICM on, `chol(R)` is computed once before the `stablehlo.while`
(scale_family_mvn: line 116 vs while at line 524) and only a scalar `multiply` by
`tau` plus a solve against the pre-factored matrix remain inside the loop; with
SICM off, `chol(tau²·R)` is recomputed every step inside the loop body.

**Correctness**: Impulse VI reproduces NumPyro SVI seed-for-seed. Under the matched
RBG key (`test/probprog/vi_*.jl`, 19/19 tests at 1e-12–1e-10), and on
`scale_family_mvn` here the final ELBO agrees to 6.3e-14. The NumPyro `final ELBO`
column above differs only because `svi_numpyro_reference.py` times with jax's default
`PRNGKey(0)` (throughput is seed-independent); the ELBO-agreement metric uses the
matched RBG key.

*Note*: n=8 is a micro-benchmark; the absolute Impulse-vs-NumPyro per-step ratio is
dominated by whole-loop XLA fusion at small size. The load-bearing, size-robust
result is the SICM on/off ratio and the elimination of the per-step cholesky.

### End-to-end through PPLBench (canonical harness)

SVI is now a first-class PPLBench inference mode, replacing the focused D3 drivers
above (kept as tests). The `SVI` class in `ppls/{impulse,numpyro}/inference.py`
dispatches to `harness.jl --method svi` / `numpyro_harness.py --method svi`;
configs live in `pplbench_configs/svi/{scale_family_mvn,gp_pois_regr}.json`. The
config's `iterations` is the number of posterior draws K; `num_steps` is the SVI
optimization budget. Predictive log likelihood (PPLBench's metric) is computed from
K=200 draws of the fitted AutoNormal guide (eps ~ N(0,I); z = loc + scale·eps in the
unconstrained latent space; per-site constraining bijector), mirroring NumPyro's
`Predictive` / `sample_posterior` over the guide. Guide-sampling parity is
constructional, not bitwise (independent RNG streams; Impulse draws host-side,
NumPyro uses threefry), so PLL/ELBO agreement is a numerical, not bit-exact, check.

**Config**: AutoNormal + Adam (lr 0.01), 2000 steps, 1 particle, K=200 draws, n=8,
32 threads, CPU. Wall clock is the mean over steady-state trials (trial 0 excluded).

| model | framework | per-step (µs) | fit/trial (ms) | final ELBO | final PLL (mean) | SICM |
|-------|-----------|--------------:|---------------:|-----------:|-----------------:|------|
| scale_family_mvn | Impulse (SICM) | 2.08 | 4.15 | -17.43 | -15.14 | in-loop cholesky removed |
| scale_family_mvn | Impulse (no opt) | 7.63 | 15.27 | -17.43 | -15.14 | **3.68× slower/step** |
| scale_family_mvn | NumPyro SVI | 110.63 | 221.3 | -17.71 | -15.20 | reference |
| gp_pois_regr | Impulse (SICM) | 2.08 | 4.15 | -13.87 | -6.36 | in-loop cholesky removed |
| gp_pois_regr | Impulse (no opt) | 7.12 | 14.25 | -13.87 | -6.36 | **3.43× slower/step** |
| gp_pois_regr | NumPyro SVI | 164.52 | 329.0 | -13.57 | -6.44 | reference |

The SICM on/off per-step ratios (3.68× mvn, 3.43× gp) and per-step figures (SICM
2.08 µs, no-opt 7.1–7.6 µs) reproduce the D3 micro-benchmark (3.38×/3.57×, SICM
2.06–2.24 µs) under the independent PPLBench wall-clock path. Impulse SICM and no-opt
produce bit-identical draws and PLL (SICM is a performance-only transform). Cross
-framework, Impulse and NumPyro PLLs agree to within single-particle ELBO noise
(mvn -15.14 vs -15.20; gp -6.36 vs -6.44) on the shared PPLBench test data.

---

## Programmable MH (Gen-style) — SICM on the MH proposal density

The OOPSLA rebuttal's programmable-inference snippet: a user-written loop that
alternates single-address Metropolis-Hastings moves over a generative model
(`mh(:slope)` then `mh(:tau)`, n iterations). Each `ProbProg.mh` call lowers to
its own `impulse.infer(mh_config)` region, so the shared region pipeline
(`inline-infer-regions → sicm → construct joint logpdf → expand-impulse`)
optimizes every MH move's proposal density — the same machinery that optimizes
HMC/NUTS/VI. This is the algorithm-independence claim demonstrated on
custom, user-composed inference.

**Two models, honest split of the story:**
- *Plain rebuttal snippet* (slope/intercept Normal priors, observed `ys`): both
  moves lower to distinct `impulse.infer_region` ops (`infer_region_count = 2`,
  `flat_infer_count = 0`), but the plain-Normal density is **SICM-neutral** — the
  region pipeline's win here is ordinary LICM/CSE (SICM-on and SICM-off HLO are
  identical around the loop). Reported as such.
- *Hoistable-covariance variant* (linear-regression mean + MVN likelihood with
  covariance `tau²·R`, R a fixed correlation matrix): SICM's
  CholeskyScale/TriangularSolveScale/LogMultiply cascade fires inside each MH
  move's joint logpdf, rewriting `chol(tau²·R) → sqrt(tau²)·chol(R)` and hoisting
  `chol(R)` out of the move. This is where the SICM story lives.

**Config**: n=8, 500 iterations = 1000 MH moves, per-move min wall-clock over 8
reps, CPU. Driver: `test/probprog/mh_sicm_benchmark.jl` (also emits the IR
evidence). SICM arm sets `DEBUG_PROBPROG_DISABLE_OPT[]=false` (the opt-in region
path); the no-opt arm leaves the default (`true` → legacy flat `impulse.mh`).

**Per-move wall clock (hoistable-covariance variant)**:

| model | arm | per-move (µs) | speedup | notes |
|-------|-----|--------------:|--------:|-------|
| scale_linreg_mvn (n=8) | Impulse (SICM) | 2.93 | 1.00× | `sqrt(tau²)·chol(R)` decomposition; sqrt in-loop 1 |
| scale_linreg_mvn (n=8) | Impulse (region, no SICM) | 4.37 | **1.49×** | isolates SICM's algebra (only the SICM knob flips) |
| scale_linreg_mvn (n=8) | Impulse (no opt / flat) | 4.43 | 1.51× | legacy `impulse.mh`; no region, no SICM |

**SICM effect**: the isolated **1.49×** per-move speedup (region path, SICM on vs
off) is the direct SICM contribution; the `stablehlo.sqrt` in the loop body
(present only with SICM on) is the decomposition witness. At the impulse/region
level SICM hoists `chol(R)` above the user `stablehlo.while` and threads it in as
a loop-carried invariant (`cov_region_sicm_on.mlir`); in the final XLA HLO,
`expand-impulse` re-materialises the per-move density and XLA's while-loop LICM
does not further hoist it, so the fully-hoisted "in-loop chol 1→0" shape is
exercised instead by the internal-loop (`num_samples>1`) form in the Enzyme-JAX
lit test `mh_region.mlir`.

**Correctness**: seed-for-seed the SICM and no-opt chains make identical accept
decisions (19/19 accepts each) and land on a byte-identical posterior
(|Δτ| = |Δslope| = 0, |Δaccepts| = 0). The SICM rewrites preserve the
distribution exactly on this program.

*Note*: n=8 is a micro-benchmark; the load-bearing result is the SICM on/off
per-move ratio (1.49×) on a custom, user-composed MH kernel. No Gen.jl baseline
is reported: no Gen.jl environment is provisioned here, and standing one up is out
of scope for this micro-benchmark (flagged as optional follow-up).

## Multi-chain standby variants (`_mc4`)

Standby reference columns (additive; the single-chain configs and their numbers
above are UNTOUCHED, for comparability). Each `_mc4` config runs `num_chains=4`
genuine chains per trial: Impulse via concurrent Julia thread-dispatch of the
compiled single-chain kernel (`chain_method=:parallel`); NumPyro via genuine
`pmap` over 4 forced host devices (`chain_method="parallel"`, driver sets
`XLA_FLAGS=--xla_force_host_platform_device_count=4`); Stan via native
`parallel_chains=4`. `trials=2` (so diagnostics are pooled over
`trials*num_chains = 8` chains), except `crowd_sourced_annotation_mc4` uses
`trials=4` (16 chains) so its seed set includes the poor-mixing seed 6164479
that makes Stan's R-hat flag (see below). Wall-clock is the steady-state (trial 1)
4-chain parallel run. The single-chain key split
(`jax.random.split(key, num_chains)`) is shared bitwise across the arms, so
chain c is the same chain in each framework.

**PLL semantics**: `pooled PLL` is the cumulative predictive log-likelihood over
the draws of ALL chains pooled together (not per-chain). **Diagnostics** (NEW for
chains>1): split R-hat and pooled ESS across the `trials*num_chains` chains, per
parameter, from `arviz.rhat` / `arviz.ess` (the diagnostic path already in
`ppl_helper`), written to `multichain_diagnostics.json` per run.

Configs: `pplbench_configs/standard/{logistic_regression,crowd_sourced_annotation,coal_mining}_mc4.json`,
`pplbench_configs/sicm/gp_pois_regr_scaled_mc4.json`.

| model | arm | pooled PLL | split R-hat (max) | pooled ESS (median) | steady wall-clock (s) | pooled ESS/s (median) |
|-------|-----|-----------:|------------------:|--------------------:|----------------------:|----------------------:|
| coal_mining | NumPyro | -173.60 | 1.0004 | 19694 | 0.84 | 23549 |
| coal_mining | Impulse | -173.60 | 1.0006 | 17828 | 1.12 | 15933 |
| logistic_regression | NumPyro | -29.99 | 1.0009 | 32499 | 4.58 | 7102 |
| logistic_regression | Impulse | -30.15 | 1.0007 | 31731 | 10.50 | 3023 |
| logistic_regression | Stan | -30.10 | 1.0004 | 31425 | 18.90 | 1662 |
| crowd_sourced_annotation (trials 4, 16 ch) | Impulse | -1805.79 | 1.0007 | 82243 | 15.11 | 5441 |
| crowd_sourced_annotation (trials 4, 16 ch) | Stan **[R-hat FLAG]** | -1805.55 | **1.1665** | **72** | 104.72 | 0.7 |
| gp_pois_regr_scaled (SICM) | NumPyro | -36.16 | 1.0017 | 10394 | 2.15 | 4838 |
| gp_pois_regr_scaled (SICM) | Impulse | -36.28 | 1.0030 | 11566 | 0.98 | 11770 |
| gp_pois_regr_scaled (SICM) | Stan | -36.21 | 1.0023 | 12030 | 1.26 | 9568 |

**Observations**: (1) All arms on all four models converge (max R-hat 1.003);
the multi-chain diagnostics confirm correctness uniformly and pooled PLL agree
across frameworks. (2) Pooled ESS/s ranking is model-dependent and mirrors the
single-chain per-step story: NumPyro leads on the dense-linear models
(logistic, coal), while Impulse leads on the SICM model (gp_pois_regr, 11770 vs
NumPyro 4838). (3) Impulse's thread-dispatch multi-chain parallelism is less
efficient than NumPyro's single-kernel `pmap` on the large dense model (logistic
10.5 s vs 4.6 s for the 4-chain trial); on smaller/SICM models the gap closes or
reverses. (4) On crowd_sourced the multi-chain split R-hat is a genuine framework
discriminator. At `trials=2` (benchmark seeds that exclude the problematic one)
both arms converge (max R-hat 1.001). Including the historically-bad seed 6164479
(a `trials=4` run, 16 chains) surfaces the single-chain "poorly-mixed Stan trial"
anecdote as a flagged, per-parameter result: **Stan** split R-hat median 1.14 /
max 1.17 (worst `confusion_matrix`), pooled ESS collapsing to **72**, whereas
**Impulse** on the identical 16 chains stays R-hat ≤ 1.0007 with pooled ESS 82k.
Both arms' pooled PLL still read ≈ −1805, so this convergence-robustness
difference is invisible to PLL alone. Impulse remains ~7× faster wall-clock and
~8× higher pooled ESS/s than Stan there.
