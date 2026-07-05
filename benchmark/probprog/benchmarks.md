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
