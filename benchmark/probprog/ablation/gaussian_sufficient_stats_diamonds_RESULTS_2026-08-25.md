# GaussianSufficientStats on posteriordb diamonds

Row for the SICM ablation figure. Pattern family `GaussianSufficientStats`,
model `diamonds` (posteriordb `diamonds-diamonds`, brms log-log price
regression, N = 5000, K = 25).

Status: COMPLETE. All five gates pass. Headline, last-trial statistic, SICM
0.128 s against No SICM 1.331 s, a 10.4x speedup; on the trial-median statistic
0.387 s against 1.648 s, 4.3x. Both are reported because they diverge; see
section 9.

One blocker was found and worked around along the way. The QR form of this rule
was not executable on CPU in this tree at all, because Reactant never registered
the `geqrf` and `orgqr` LAPACK runtime symbols. Section 8 has the diagnosis, the
four-line upstream patch, and why this session did not apply it directly.

**RESOLVED 2026-08-25 (update).** The four registrations now live permanently in
`Reactant.jl/src/stdlibs/LinearAlgebra.jl`, in a QR block just after the getrf
entries of `TracedLinearAlgebra.__init__`. This is pure Julia, so no rebuild was
needed and the binary is untouched. The local workaround in
`sicm/diamonds_regression.jl` has been neutralized: the
`register_qr_lapack_symbols!()` call in `setup` is commented out, and the
function is retained only so it can be restored in one line if the permanent
registration ever regresses. Verified end to end by re-running the diamonds
smoke with the local call disabled: it compiles and both arms run
(`probe/sicm_smoke_diamonds`, run dir `outputs/2026-08-25_14:38:18`, trial times
0.54 / 0.01 s and 0.58 / 0.01 s, no "Symbols not found"). Consequently the
header claim in `sicm/linreg_qr_stats.jl` that the rule executes on CPU via
LAPACK symbols registered in `TracedLinearAlgebra.__init__` is now TRUE, where
Section 8 correctly recorded it as false at the time of writing.

---

## 1. The pattern already existed

No pattern was written in this session and the compiler was NOT rebuilt.

The rule is implemented in
`/mnt/sbrantq/probprog/oopsla/Enzyme-JAX/src/enzyme_ad/jax/Passes/SampleInvariantCodeMotion.cpp`
lines 2394-3023, committed in `0c6a4a61c` (2026-08-20), as a family of four
patterns:

| pattern | line | form | registration |
|---|---|---|---|
| `GaussianSufficientStatsHLO` | 2890 | QR | aggregate `SicmPatterns` set, default |
| `GaussianSufficientStatsReduceHLO` | 2914 | QR | aggregate set, `reduce{add}` root |
| `GaussianSufficientStatsGramHLO` | 2975 | Gram | opt-in, `PatternBenefit(2)`, f64 gate |
| `GaussianSufficientStatsGramReduceHLO` | 3000 | Gram | opt-in, `reduce{add}` root |

The Gram form is behind the pass option `gram-form-sufficient-stats`
(`Passes.td:770`) and is registered at a higher benefit so it pre-empts the QR
form on the shared match. Two roots per form exist because `canonicalize` runs
before `sicm` and rewrites `reduceSumToScalar`'s dot-against-ones into
`stablehlo.reduce{add}`.

Design note: `/mnt/sbrantq/probprog/oopsla/GRAM_RULE_DESIGN.md` (744 lines).

### Lit tests (pre-existing, both pass)

    //test/lit_tests:impulse_sicm_gaussian_sufficient_stats.mlir.test
    //test/lit_tests:impulse_sicm_gaussian_sufficient_stats_gram.mlir.test

Sources `test/lit_tests/impulse/sicm/gaussian_sufficient_stats.mlir` and
`gaussian_sufficient_stats_gram.mlir`. The QR file already contains every case
the diamonds work would have needed, so no new lit tests were added; duplicates
would be noise.

| test function | role |
|---|---|
| `@linreg_qr` | fires on the emitted `dot_general(kernel, ones)` shape |
| `@linreg_qr_reduce` | fires on the canonicalized `reduce{add}` root |
| `@neg_x_dependent` | X sample-dependent, must not fire |
| `@neg_no_square` | non-residual reduce, must not fire |
| `@neg_small_n` | shape guard N >= 2D declines |

Full Enzyme-JAX lit suite on this tree: 1217 of 1219 pass. The two failures,
`diffrules_enzymexla_gpu_wrapper_fwd` and `parallel-lower-noreturn`, are
unrelated to SICM and pre-existing from a sibling session's uncommitted
`LowerEnzymeXLABLAS.cpp` / `LowerEnzymeXLALapack.cpp` / `StablehloToLapack.cpp`
work.

---

## 2. Matched form

Read off the real emitted HLO and confirmed by running the pass, not from an
imagined canonical form.

Root is the scalar event summation, either `dot_general(kernel, ones_N)` or the
canonicalized `reduce{add}`. Walking up the operand chain the matcher requires,
in order:

1. any number of scalar-affine wrappers (`mul`, `div`, `add`, `sub` against an
   effectively scalar operand), peeled by `stripAffineScalarWrappers` (2571);
2. a square, either `mul(W, W)` or `pow(W, 2)`, via `matchSquare` (2632);
3. an optional effectively-scalar factor such as the `1/sigma` division, via
   `peelScalarFactor` (2644);
4. a residual `subtract` one of whose operands is **directly** a `dot_general`
   of a rank-2 sample-invariant design against a rank-1 sample-dependent
   vector, the other operand being a sample-invariant rank-1 response, via
   `matchGaussianResidual` (2683).

Preconditions:

- **structure**: the chain above;
- **dependence**: X and y sample-invariant, beta sample-dependent;
- **shape guard**: `N >= 2 * D`, static (2761);
- **sign**: none, and the QR form introduces no root of a sample-dependent
  quantity;
- **element type**: the Gram form additionally requires f64 (`gramElemTypeOk`,
  2971); the QR form has no such gate.

The sample-dependent parameter vector may be produced by `stablehlo.concatenate`,
which is what a model with two different priors on different coefficient blocks
emits. Verified by probe: the matcher only requires rank 1 and sample
dependence, so `vcat(intercept, b)` is accepted.

---

## 3. Coverage gap: two faces of one missing walk-up step

Both faces fail at the same place. `matchGaussianResidual`
(`SampleInvariantCodeMotion.cpp:2683-2740`, decisive line 2691,
`auto dot = maybeMean.getDefiningOp<stablehlo::DotGeneralOp>();`) requires an
operand of the residual `subtract` to be **directly** a `dot_general`. Any op
interposed between the subtract and the mean contraction makes the rule
decline. Both faces are reported, not patched, because patching requires a
rebuild of the shared binary.

### Face 1: broadcast-scalar shift, `mu = alpha .+ X * beta`

Emits `subtract(y, add(dot_general(Xc, b), broadcast(alpha)))`. The walk-up hits
the `add` and stops.

Probed at the real diamonds dims. `@diamonds_split_intercept` produced zero
`enzymexla.linalg.qr` and kept the entire length-5000 chain, including the
`dot_general` at `5000x24`:

    %2 = stablehlo.dot_general %arg5, %1, contracting_dims = [1] x [0] : (tensor<5000x24xf64>, tensor<24xf64>) -> tensor<5000xf64>
    %4 = stablehlo.add %2, %3 : tensor<5000xf64>          <-- blocks here
    %5 = stablehlo.subtract %arg6, %4 : tensor<5000xf64>

In the tree: `standard/robust_regression.jl:14` writes exactly this spelling,
though its likelihood is StudentT, so the Gaussian rule would not apply to it
regardless.

Protocol irony worth recording. The form that MISSES the rule is Stan and brms's
own performance-oriented centering transform, which splits the intercept out so
`normal_id_glm_lpdf` can take it as a separate argument. The plain statistician
translation, one design matrix with an intercept column, is the one that fires.

### Face 2: reshape or slice between subtract and dot

`mu = (X * reshape(beta, k, 1))[:, 1]` and `mu = reshape(X * reshape(beta, k, 1), n)`
emit `subtract(y, reshape(dot_general(X, beta_2d)))`. The walk-up hits the
`reshape` and stops.

Probed at the real diamonds dims, `@reshape_slice_mean` produced zero
`enzymexla.linalg.qr`:

    %2 = stablehlo.reshape %1 : (tensor<25xf64>) -> tensor<25x1xf64>
    %3 = stablehlo.dot_general %arg5, %2, contracting_dims = [1] x [0] : (tensor<5000x25xf64>, tensor<25x1xf64>) -> tensor<5000x1xf64>
    %4 = stablehlo.reshape %3 : (tensor<5000x1xf64>) -> tensor<5000xf64>   <-- blocks here
    %5 = stablehlo.subtract %arg6, %4 : tensor<5000xf64>

This face costs real coverage today. The following existing benchmark models
have an iid-Gaussian likelihood over a design-matrix mean, so the rule should
fire on them, and does not, subject to each config's `N >= 2D`:

| model | lines | Gaussian likelihood sites blocked |
|---|---|---|
| `standard/correlated_prior_regression.jl` | 26, 28 | 1 |
| `standard/multi_task_regression.jl` | 34-35, 40-41, 46-47, 52-53 | 4 |
| `standard/multi_task_regression_k10.jl` | 16, 20, 24, 28, ... | 10 |

Two further models use the reshape spelling but have a `MultiNormal` likelihood,
so the Gaussian rule is inapplicable to them regardless and they are NOT
affected: `sicm/phylogenetic_regression.jl:62,72` and
`sicm/linear_mixed_model.jl:34` (the latter is an `EigenLift` instance).

Models that use the direct spelling and do fire: `sicm/linreg_qr_stats.jl:39-40`,
`sicm/radon_qr_stats.jl:49-50`, and `sicm/diamonds_regression.jl:58`.

### Suggested fix for a future session

Extend the walk-up in `matchGaussianResidual` to look through shape-only ops
(`reshape`, and a full-extent `slice`) and to absorb a broadcast-scalar shift by
treating it as an extra design column of ones, that is
`||y - X beta - alpha 1||^2 = ||y - [1, X] [alpha; beta]||^2`. Both faces then
collapse to the existing match. This is a matcher change only; the emission and
the preconditions are unchanged.

---

## 4. Numerical honesty

Diamonds is a high-R^2 instance, which is exactly the regime
`GRAM_RULE_DESIGN.md` section 3.5 flags as dangerous for the Gram form. The
design note argued this in the abstract; these are the numbers on the real
instance.

Measured on the actual augmented design matrix `X_aug = [1, Xc]` in f64:

| quantity | value |
|---|---|
| N, K | 5000, 25 |
| `y'y` | 3.084206e+05 |
| RSS at OLS | 7.416944e+01 |
| ratio `y'y / RSS` | 4.158e+03 |
| `cond(X)` | 714.6 |
| `cond(X'X)` | 5.106e+05 |
| OLS sigma | 0.12210 |
| OLS R^2 | 0.98561 |

`cond(X'X)` is `cond(X)^2` to three figures, the textbook statement that the
normal-equations form squares the conditioning. That is the whole reason the QR
form is the shipped default.

`max |logpdf_form - logpdf_direct|` in nats, 400 draws per regime, draws taken
as `beta_ols + scale * sigma * R^-1 z`:

| regime | Gram | QR |
|---|---|---|
| at OLS | 3.235e-08 | 1.906e-12 |
| posterior scale | 5.300e-08 | 7.912e-11 |
| wide | 6.757e-08 | 3.441e-10 |

The QR form is about two orders of magnitude more accurate than Gram on this
instance, and both are far below anything that could perturb an MCMC
trajectory. The figure row measures the default QR form.

Reproduce: the block is a short numpy script over
`DiamondsRegression.generate_data`; statistics are `G = X'X`, `Xty = X'y`,
`yty = y'y` for the Gram form and a thin `numpy.linalg.qr` with `R = Q1'X`,
`c = Q1'y`, `rem = ||y - Q1 c||^2` for the QR form.

---

## 5. IR evidence, compiler side

All three probes were run through the real pass with the freshly built
`enzymexlamlir-opt`, at the real diamonds dims, with
`--enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats[_gram]"
--transform-interpreter --enzyme-hlo-remove-transform`.

**QR form on the ones-column / concat-beta model (the figure row's form).**
Fires. Hoistable invariant statistics, all functions of the `X` and `y` block
arguments:

    %Q, %R, %info = enzymexla.linalg.qr %arg5 : (tensor<5000x25xf64>) -> ...
    %5 = stablehlo.dot_general %Q, %arg5, contracting_dims = [0] x [0] -> tensor<25x25xf64>   // R = Q1'X
    %6 = stablehlo.dot_general %Q, %arg6, contracting_dims = [0] x [0] -> tensor<25xf64>      // c = Q1'y
    %7 = stablehlo.dot_general %Q, %6,    contracting_dims = [1] x [0] -> tensor<5000xf64>    // Q1 c
    %8 = stablehlo.subtract %arg6, %7
    %9 = stablehlo.dot_general %8, %8, contracting_dims = [0] x [0] -> tensor<f64>            // rem

and the in-loop residual is K-dimensional throughout:

    %10 = stablehlo.dot_general %5, %4,  contracting_dims = [1] x [0] -> tensor<25xf64>       // R beta
    %11 = stablehlo.subtract %6, %10                                                          // c - R beta
    %12 = stablehlo.dot_general %11, %11, contracting_dims = [0] x [0] -> tensor<f64>

**Gram form on the same model** (opt-in variant, recorded as a side observation,
NOT the figure row). Fires, emitting only `dot_general` and no LAPACK op:

    %5 = stablehlo.dot_general %arg5, %arg5 -> tensor<25x25xf64>   // G   = X'X
    %6 = stablehlo.dot_general %arg5, %arg6 -> tensor<25xf64>      // Xty = X'y
    %7 = stablehlo.dot_general %arg6, %arg6 -> tensor<f64>         // yty = y'y
    ... in-loop yty - 2 beta'Xty + beta'G beta ...

A wall-clock Gram arm is not reported. Selecting it requires
`Reactant.Compiler.DEBUG_PROBPROG_GRAM_STATS[]`, which `harness.jl` does not
expose as a CLI flag and `ppls/impulse/inference.py` does not pass, so measuring
it would mean editing shared harness files that the sibling campaign is using.
The numerical comparison in section 4 is the part that matters and it is
recorded.

**Split-intercept and reshape forms.** Do not fire; see section 3.

---

## 6. Model and data

### Data

`pplbench/pplbench/models/real_data/diamonds.json`, 2344779 bytes,
md5 `b7e51f0db63b8bd047f029b4990af235`.

Verbatim copy of the posteriordb payload (`N`, `K`, `Y`, `X`, `prior_only`) with
a `_provenance` string as the first key. Source is
`posterior_database/data/data/diamonds.json.zip`, posterior `diamonds-diamonds`,
model `posterior_database/models/stan/diamonds.stan`, generated with brms
2.10.0. Nothing is simulated and nothing is subsampled. Column 0 of `X` is the
intercept column of ones, asserted at load.

Regenerated byte-identically by
`ablation/sicm_roster/build_diamonds.py`, which mirrors the existing
`build_nyc_submap.py` convention.

### Model

`sicm/diamonds_regression.jl` and `pplbench/pplbench/models/diamonds_regression.py`.

The Stan file is

    Intercept ~ student_t(3, 8, 10)
    b         ~ normal(0, 1)                 (Kc = 24)
    sigma     ~ student_t(3, 0, 10) T[0,]
    Y         ~ normal_id_glm(Xc, Intercept, b, sigma)

with `Xc` the 24 non-intercept columns of `X`, each centered at its own column
mean. The benchmark folds the intercept back into the design matrix. With
`X_aug = [1, Xc]` and `beta = [Intercept; b]` the mean is the single product
`X_aug beta`, which is the identical likelihood with identical parameters and
identical priors. The centering is done in the Python model class, mirroring
Stan's `transformed data` block; `max |column mean|` of the centered block is
1.27e-14 and `cond(X_aug) = 714.6`.

One prior substitution, documented in both the model file and
`real_data/README.md`: ProbProg has no half-student-t, so `sigma` uses
`HalfCauchy(10)`, the half student-t with one degree of freedom rather than
three, at the same scale and on the same support. With N = 5000 at R^2 = 0.986
the residual SD is pinned by the data, so the choice does not move the
posterior. The `Intercept` and `b` priors are carried over exactly.

The mean is written `X * beta` with a 1-D `beta` on purpose, so that face 2 of
the coverage gap does not silently disable the rule. The model file carries a
comment saying so.

### Pre-flight checks (all green, before any timed run)

- all four configs validate against pplbench's own jsonschema;
- model class and impulse impl resolve exactly as `model_helper` and
  `ppl_helper` do it; attrs `{N: 5000, K: 25, Kc: 24}` match the impl
  constructor;
- `generate_init_params` returns `beta (25,)` and `sigma (1,)`;
- the Python PLL evaluator runs and gives 3432.4 at the OLS parameters;
- the Julia model parses and defines `setup`, `build_constraint`,
  `extract_samples`.

One defect these pre-flight checks did NOT catch, recorded because it is a
methodology lesson. The first smoke attempt failed immediately with
`KeyError: 'n'` at `pplbench/lib/model_helper.py:48`. The dataset attributes had
been named `N`, `K`, `Kc`, but pplbench's own `simulate_data` reads
`attrs["n"]`, and `ppl_helper` splats the whole attrs dict into both the PPL
implementation's constructor (line 132) and `generate_init_params` (line 170).
The pre-flight check had called `generate_data` directly and so bypassed
`simulate_data` entirely, which is exactly the call that failed. Fixed by
renaming the attributes to the lowercase `n`, `k`, `kc` that every other model
uses, and by re-verifying through `model_helper.simulate_data` itself rather
than around it. The JSON data file keeps posteriordb's own `N` and `K` key
names; only the xarray attribute names changed. Cost: one failed 1-second smoke
run, which is precisely what the smoke stage exists to absorb.

### Files created or edited

Created:

    pplbench/pplbench/models/real_data/diamonds.json
    pplbench/pplbench/models/diamonds_regression.py
    pplbench/pplbench/ppls/impulse/diamonds_regression.py
    sicm/diamonds_regression.jl
    pplbench_configs/sicm_ablation/diamonds.json
    pplbench_configs/probe/sicm_smoke_diamonds.json
    pplbench_configs/probe/sicm_ir_diamonds_sicm.json
    pplbench_configs/probe/sicm_ir_diamonds_nosicm.json
    ablation/sicm_roster/build_diamonds.py
    ablation/sicm_roster/run_diamonds.sh
    ablation/gaussian_sufficient_stats_diamonds_RESULTS_2026-08-25.md   (this file)

`run_diamonds.sh` is deliberately separate from `run_batch.sh`, which owns the
four-cell figure batch and is in use by a concurrent session. It runs one stage
at a time (`smoke`, `ir`, `timed`), records the binary md5 and `/proc/loadavg`
before and after each stage exactly as `run_batch.sh` does, and its `timed`
stage refuses to start while any other `pplbench` or `harness.jl` process is
alive, since a second uncapped run would contaminate both. That refusal guard
was verified against a real concurrent run of the sibling's batch and correctly
refused.

Edited:

    pplbench/pplbench/models/real_data/README.md   (diamonds section, subsample bullet, substitution note)

No file belonging to the sibling campaign was edited. The timed cell runs via
the `CELLS` environment override that `ablation/sicm_roster/run_batch.sh`
already honors, so none of its scripts needed changing.

---

## 7. Binary

    /mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so
    md5   21a6bd08b37b54d2b8a20d8924a2bbb2
    built 2026-08-24 14:59

This is the SAME binary every other row of the ablation figure ran on. The
pattern landed in `0c6a4a61c` on 2026-08-20, four days before this binary was
built, and `strings` confirms `GaussianSufficientStatsHLO`,
`GaussianSufficientStatsGramHLO`, the Reduce variants and the transform ops
`SicmGaussianSufficientStatsPatterns` / `...GramPatterns` are all compiled in.
No rebuild was performed or needed, so the diamonds row does not break the
single-build property of the campaign.

---

## 8. Blocker found and fixed: the QR runtime symbols were never registered

The first smoke run of the SICM arm failed at compile time:

    Compiling... JIT session error: Symbols not found: [ enzymexla_lapack_dgeqrf_ ]
     lookupError Failed to materialize symbols: { (enzymejitdl_0, { entry }) }
    JIT session error: Symbols not found: [ enzymexla_lapack_dorgqr_ ]
    ERROR: LoadError: CompilationError: MLIR pass pipeline "impulse" failed

The rule fires correctly and emits `enzymexla.linalg.qr`, which lowers to calls
to `enzymexla_lapack_dgeqrf_` and `enzymexla_lapack_dorgqr_`. Reactant's
`TracedLinearAlgebra.__init__` in `src/stdlibs/LinearAlgebra.jl` maps its LAPACK
and BLAS runtime symbols with `MLIR.API.EnzymeJaXMapSymbol`, covering gtsv,
getrf, gesvd, gesdd, gesvj, syevd, syrk, trmm, symm, potrf and trsm. It has NO
geqrf or orgqr entry, so those two calls never resolve and the entire `impulse`
pass pipeline fails.

Consequence worth flagging beyond this row: the QR form of
`GaussianSufficientStats` was NOT executable on CPU in this tree. The existing
`sicm/linreg_qr_stats.jl` asserts in its own header comment that the rule
"executes on CPU via the Fortran LAPACK dgeqrf_/dorgqr_ runtime symbols
registered in Reactant's TracedLinearAlgebra.__init__ (same mechanism syevd
uses)". That registration does not exist. Any previously claimed CPU result from
`linreg_qr_stats.jl` or `radon_qr_stats.jl` should be re-checked against this.

**Recommended upstream patch**, four lines in the symbol table of
`Reactant.jl/src/stdlibs/LinearAlgebra.jl`, alongside the existing potrf and
trsm entries:

    # QR factorization (geqrf / orgqr)
    (BLAS.@blasfunc(sgeqrf_), :enzymexla_lapack_sgeqrf_),
    (BLAS.@blasfunc(dgeqrf_), :enzymexla_lapack_dgeqrf_),
    (BLAS.@blasfunc(sorgqr_), :enzymexla_lapack_sorgqr_),
    (BLAS.@blasfunc(dorgqr_), :enzymexla_lapack_dorgqr_),

This session did NOT apply that patch, because a concurrent session has
uncommitted changes in exactly that file and exactly that list (it is adding the
potrf and trsm entries). Instead the same four registrations are made locally in
`sicm/diamonds_regression.jl` via `register_qr_lapack_symbols!()`, called at the
top of `setup`. That is additive, touches nothing shared, and is functionally
identical because `EnzymeJaXMapSymbol` is a runtime mapping. It is a workaround
and the permanent home is the file above.

**Applied 2026-08-25.** The patch above is now in
`Reactant.jl/src/stdlibs/LinearAlgebra.jl` (QR block immediately after the
getrf entries), user-authorized once the concurrent edits to that file had
landed. The local `register_qr_lapack_symbols!()` call is commented out and the
permanent path is verified by the diamonds smoke; see the update note at the top
of this file. Nothing in this row's measured numbers changes: the registration
is a runtime symbol mapping either way, and the row was measured with the local
registration active.

No rebuild of `libReactantExtra.so` was required. The fix is pure Julia.

---

## 9. Timed two-arm cell

    ./ablation/sicm_roster/run_diamonds.sh timed diamonds1

Config `pplbench_configs/sicm_ablation/diamonds.json`: 500 iterations, 200
warmup, 2 trials, seed 12345, step size 0.05, max tree depth 10, step size and
mass matrix adaptation both on, threads uncapped, arms `SICM` and `No SICM`.
Output dir `outputs/2026-08-25_03:19:53`.

The expectation recorded before the run was 2x to 20x, on the reasoning that the
flop ratio is about `N / K = 200` but at these sizes wall clock is dominated by
overhead and by the elimination of reverse-pass work. The measurement landed
inside that band and is reported as measured. The instance was not retuned.

| arm | compile | trial 0 | trial 1 | last trial | trial median |
|---|---|---|---|---|---|
| SICM | 29.76 s | 0.645 s | 0.128 s | **0.128 s** | **0.387 s** |
| No SICM | 31.33 s | 1.964 s | 1.331 s | **1.331 s** | **1.648 s** |

| statistic | speedup |
|---|---|
| last trial | **10.4x** |
| trial median | **4.3x** |

Both statistics are reported because they diverge, the same way the sibling's
radon row diverges (43x against 20x). The cause here is visible in the table.
Trial 0 carries first-call warm-up in both arms, and it dominates a two-trial
median. The SICM arm's trial 0 is 5.0x its trial 1, while the No SICM arm's
trial 0 is only 1.5x its trial 1, so the shared fixed overhead is a much larger
fraction of the faster arm and the median statistic compresses the ratio. The
last-trial number is the steady-state one. Which convention the figure adopts is
a campaign-wide decision, not a per-row one.

Caveat on precision, stated rather than hidden. This row's absolute times are
small, 0.128 s for the fast arm against 7.7 s and 377.1 s for the existing GP
rows. A tenth-of-a-second measurement has correspondingly less headroom above
timer and scheduler noise, and the same caveat applies to the sibling's radon
row at 0.15 s. If the campaign wants these fast rows on the same footing as the
slow ones, the honest fix is more trials or more iterations, not a different
statistic.

Machine conditions: loadavg 2.30 before and 2.64 after, no other pplbench or
harness process alive (the runner's collision guard verified this and would have
refused otherwise). Binary md5 `21a6bd08b37b54d2b8a20d8924a2bbb2` before and
after, unchanged.

### Gate results

**1. Chains move and mix. PASS.**

| arm | n_eff [min, med, max] | Rhat [min, med, max] | unique beta draws/chain | unique sigma draws/chain |
|---|---|---|---|---|
| SICM | 145.2, 573.8, 1035.8 | 0.9971, 1.0011, 1.0150 | 7500 / 7500 | 300 / 300 |
| No SICM | 169.5, 574.3, 1125.1 | 0.9981, 1.0014, 1.0077 | 7500 / 7500 | 300 / 300 |

No stuck chain, no Rhat anywhere near the 1e3 invalidation threshold, every
drawn value distinct.

**2. Arm posterior agreement. PASS.**

Adaptation is on, so the two arms take different trajectories and pointwise
sample equality is not expected. `postcheck.py`'s pointwise `max |arm0 - arm1|`
of 1.617 on beta is the wrong statistic here; for two independent chains from a
posterior with sd about 0.35 a pointwise max of that size over 7500 comparisons
is ordinary. The correct check is agreement of posterior moments relative to
Monte Carlo error, computed with arviz ESS on the post-warmup draws:

| quantity | value |
|---|---|
| beta posterior-mean agreement, max z | **1.30** |
| beta posterior-mean agreement, median z | 0.42 |
| sigma posterior-mean agreement, z | 0.69 |
| max relative difference in posterior sd | 16.5 % |

where `z = |mean_SICM - mean_NoSICM| / sqrt(sd^2/ESS + sd^2/ESS)`. Every
coordinate is within 1.3 standard errors, so the arms agree.

Both arms also recover the independent OLS reference, which is a check on the
model rather than on the arms:

| quantity | SICM | No SICM | OLS |
|---|---|---|---|
| intercept, beta[0] | 7.78806 | 7.78805 | 7.78800 |
| sigma | 0.122864 | 0.122811 | 0.122100 |

**3. IR evidence. PASS.** See section 5 for the compiler-side probes and the
table below for the dumps taken from the real runs
(`hlo/diamonds_{sicm,nosicm}.mlir`, from `probe/sicm_ir_diamonds_*`).

| | SICM | No SICM |
|---|---|---|
| pipeline | `sicm{gram-form-sufficient-stats=false max-iterations=10}` | `sicm{... max-iterations=0}` |
| `(5000x25, 5000) -> 25` | 1, hoisted (`c = Q1'y`) | 0 |
| `(5000x25, 25x5000) -> 25x25` | 1, hoisted (`R = Q1'X`) | 0 |
| `(5000x25, 25) -> 5000` | 1, hoisted (`Q1 c` for `rem`) | 0 |
| `(25x5000, 25) -> 5000` in loop | 0 | **3** (forward mean contraction) |
| `(5000, 25x5000) -> 25` in loop | 0 | **3** (reverse-pass adjoint) |
| `(25x25, 25) -> 25` in loop | **3** (`R beta`) | 0 |
| `(25, 25x25) -> 25` in loop | **3** (its adjoint) | 0 |
| length-5000 elementwise ops | **4** | **33** |
| CPU LAPACK custom calls | 6 | 4 |

The O(N K) contraction count goes from six in-loop instances to three hoisted
one-time instances, the in-loop work becomes six K-dimensional dots, and the
length-5000 elementwise chain drops from 33 ops to 4. The three in-loop
`(5000, 25x5000) -> 25` adjoints on the No SICM side are the direct evidence for
the pre-AD claim, that running SICM before differentiation collapses the reverse
pass to K-dimensional work as well. The two extra CPU LAPACK custom calls on the
SICM side are the geqrf and orgqr from the emitted QR.

**4. On-chain numerical agreement. PASS.** Section 4's statistic recomputed at
every post-warmup draw of the actual chains rather than on synthetic draws,
`|logpdf_form - logpdf_direct|` in nats:

| arm | draws | Gram max | QR max | Gram median | QR median |
|---|---|---|---|---|---|
| SICM | 600 | 4.681e-08 | **5.415e-11** | 3.259e-08 | 1.398e-11 |
| No SICM | 600 | 4.675e-08 | **6.650e-11** | 3.214e-08 | 1.399e-11 |

The figure row measures the QR form, whose worst on-chain deviation is 5.4e-11
nats. The Gram column is what the opt-in variant would have cost on the same
draws, about three orders of magnitude worse, consistent with the synthetic
estimate in section 4 and with the design note's reasoning.

**5. Single-build discipline. PASS.** md5 `21a6bd08b37b54d2b8a20d8924a2bbb2`
recorded before and after every stage, identical throughout, and identical to
the binary the other figure rows ran on. Loadavg recorded per stage. Full
records in `ablation/sicm_roster/logs_diamonds/diamonds1/meta.txt`.

### No-regression check

`probe/sicm_smoke_car_scotland` was re-run on the same binary after all of this
session's changes. Both arms complete, no failures, and the two arms are
bit-identical on PLL (`-126.5585` on trial 0 and `-128.9702` on trial 1 in both
SICM and No SICM). Nothing this session did perturbs an existing model, which is
expected because every change was a new file; the only edit to an existing file
was documentation (`real_data/README.md`).

Note on the brief's wording. It asked for a `hierarchical_mvn` smoke, but no
`probe/sicm_smoke_hierarchical_mvn.json` exists in the tree; the smoke configs
present are `car_nyc`, `car_scotland`, `diamonds`, `lmm_radon` and `phylo_pois`.
`car_scotland` was used instead and is the cheapest of them.

---

## 10. Figure row (NOT written by this session)

`paper/figures/sicm_ablation_bar_data.json` was deliberately NOT edited. Its
`_provenance.note` already records that the model roster is under revision as of
2026-08-25, and a sibling session owns that file. The row below is offered for
whoever lands the roster, in the file's existing schema.

Under the file's stated convention, last-trial sampling times:

    {"pattern": "GaussianSufficientStats", "label": "Diamonds Regr.",
     "size": "$N{=}5000,\\,K{=}25$",
     "sicm_s": 0.13, "nosicm_s": 1.33, "speedup": "10×"}

Under the trial-median convention, if the campaign switches to it:

    {"pattern": "GaussianSufficientStats", "label": "Diamonds Regr.",
     "size": "$N{=}5000,\\,K{=}25$",
     "sicm_s": 0.39, "nosicm_s": 1.65, "speedup": "4.3×"}

Two notes for whoever lands it. The existing rows carry one decimal place, which
at 0.1 s would quantize this row's fast arm badly, so two decimals are used
here; if the file must stay at one decimal then the sensible alternative is to
report this row in milliseconds or to raise its iteration count. And the
`speedup` strings in the file use no decimal at or above 10 (`49×`, `50×`) and
one decimal below 10 (`3.9×`), which is what `10×` and `4.3×` follow.

---

## 11. Proposed paper text

For the Design-section pattern catalog in `paper/Design.tex`, after the
`EigenLift` paragraph. NOT written to the paper by this session; `paper/` was
not touched. If it is used, house rules say it must go in wrapped in `\draft{}`.

    \paragraph{Sufficient-statistics rewriting (\textsc{GaussianSufficientStats})}
    In a Gaussian linear model the likelihood touches the data only through the
    residual sum of squares $\lVert y - X\beta \rVert^2$, where the design
    matrix $X$ and the response $y$ are sample-invariant while the coefficients
    $\beta$ are sampled, so nothing inside the residual hoists on its own.
    SICM rewrites that sum through a thin QR factorization $X = Q_1 R$, using
    $\lVert y - X\beta \rVert^2 = \lVert c - R\beta \rVert^2 + r$ with
    $c = Q_1^\top y$ and $r = \lVert y - Q_1 c \rVert^2$ both sample-invariant
    and hoisted.
    The per-iteration cost falls from $O(nd)$ plus several length-$n$
    elementwise kernels to $O(d^2)$, and because SICM runs before
    differentiation the reverse pass collapses to $d$-dimensional work as well.

Notes for whoever lands it:

- Three sentences, no colon inside any sentence, no dashes, matching the
  `OuterProdScale` and `EigenLift` paragraphs. Uses $n$ for observations and $d$
  for coefficients, which is the notation the surrounding catalog already uses
  for dimensions.
- This is the first catalog entry that is not a factorization rewrite, so if the
  section wants to mark that, the natural place is a clause in the surrounding
  prose rather than a fourth sentence here.
- The count wording flagged in the commented-out block at `Design.tex:239-244`
  ("five scale-decomposition patterns") needs another look if this entry lands,
  since it would no longer be five.
- Optional portability sentence, only if the section wants it, otherwise leave
  it out because it opens a numerics discussion the catalog does not otherwise
  have. "A normal-equations variant that forms $X^\top X$ directly is available
  for targets without a QR primitive, at the cost of squaring the conditioning
  of the design."
- Suggested citation anchors if the paragraph wants them, in the style of the
  neighbouring paragraphs, are the standard least-squares QR treatment
  (Golub and Van Loan) and the brms or posteriordb provenance of the evaluated
  instance.
