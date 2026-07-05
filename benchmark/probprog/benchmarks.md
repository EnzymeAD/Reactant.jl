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
