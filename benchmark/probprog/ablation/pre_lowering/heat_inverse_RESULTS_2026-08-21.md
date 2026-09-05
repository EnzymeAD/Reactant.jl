# Heat-Equation Inverse: Pre-Lowering Ablation Results (2026-08-21, rev. 2)

Standard-passes-only axis (rev. 2; supersedes both the July unroll-workaround
numbers and the earlier region-bundled arm definition in rev. 1 of this file).
20 NUTS samples, step 0.02, max_tree_depth 5, adaptation off, CPU
(`CUDA_VISIBLE_DEVICES=""`), medians of 5 runs, all table/context cells from
ONE controlled batch on an idle machine. Binary: libReactantExtra.so mtime
2026-08-21 01:29 (includes `cholesky_licm` and the post-AD
`stablehlo-to-lapack` pass; every arm runs LAPACK-backed dpotrf/dtrsm — do
not mix with pre-LAPACK numbers).

## Arms (tab:prelowering; NO SICM machinery on either side)

- **A (No Pre-Opt)**: flat expand path (`DEBUG_PROBPROG_DISABLE_OPT=true`) +
  `disable_licm_optimization_passes=true` (strips the `*_licm` pattern subset
  from the standard passes, pre- and post-AD).
- **B (Pre-Opt)**: flat expand path + full standard passes.
- The ONLY cross-arm difference is the LICM pattern subset. The region/SICM
  pipeline is off in both. Arm R (region pipeline, `sicm{max-iterations=0}`)
  is production-pipeline context, not a table arm.
- SICM-off is EXPLICIT and VERIFIED, not incidental (user directive
  2026-08-21): M = I + dt*k*L is exactly CholeskyEigenLift's target, so this
  model (unlike the matmul rows) would trigger SICM if enabled. Checks:
  (a) flat path structurally never runs the sicm pass; (b) DISABLE_SICM=true
  set in both arms; (c) zero `sicm_*` patterns in the standard pass list
  (EnzymeXLA.cpp default list builder); (d) syevd (EigenLift's signature op)
  = 0 in every measured cell's final IR, all 8 cells of the final batch.

## Table numbers (final controlled batch)

| Size | Pre-Opt (B) | No Pre-Opt (A) | Speedup |
|---|---|---|---|
| D=100, T=100 | 489 ms | 3392 ms | 6.9x |
| D=300, T=100 | 2748 ms | 21674 ms | 7.9x |

Speedup grows with model size at fixed T. Run-to-run spread ~ +-15%;
re-freeze in the final single-build sweep.

## Context cells (same batch)

| Size | R: region, LICM on | Roff: region, LICM off |
|---|---|---|
| D=100, T=100 | 222 ms | 2982 ms |
| D=300, T=100 | 2033 ms | 20327 ms |

Reads: (1) Roff ~= A — with the factorization stuck in the loop, the path is
irrelevant; nothing downstream recovers it. (2) The production pipeline
(region form on top of the standard passes) widens the total pre-lowering gap
to 13.4x / 10.0x; the region contribution is Impulse-specific restructuring,
hence excluded from the standard-passes table. (3) Region-vs-region LICM-only
would read 13.4x / 10.0x but is not monotone in D; the flat axis is.

## Exploratory matrix (mixed batches; ordering info only)

Flat axis A/B medians: D=60,T=50: 635/132 = 4.8x; D=60,T=100: 976/186 = 5.2x.
Old D=300,T=30 config (region-bundled arms, superseded): 8281/1131 = 7.3x.
Flop bounds T*D/(D+T): 50 (D=100,T=100), 75 (D=300,T=100); measured 6.9/7.9.
Compression cause: eliminated op (dpotrf) is threaded BLAS-3, remaining
per-step solves are memory-bound BLAS-2, so wall-clock ratios sit well below
flop ratios (the SICM-on-VI rerun saw the same effect when LAPACK landed).

## Correctness

Same seed => identical accept decisions in both table arms at every checked
scale. Samples: bit-identical at scale 0 (max |diff| = 0.0); max |diff| =
7.4e-12 at scale 1 for the final flat A/B pair (ULP floor from LAPACK/fusion
reordering). July NumPyro parity (density rel diff <= 2e-14, lockstep
<= 5e-12) covers the model math; re-run the NumPyro lockstep at
number-freeze time since NUTS-lowering edits are in flight elsewhere.

## IR evidence (final IR, shape-classified dpotrf custom_calls, D=100)

- Arm A: 22 potrf sites total — **20 inside leaf (time) whiles** (forward +
  reverse instances), 1 in outer NUTS while, 1 top-level (init).
- Arm B: **2 potrf total** — 1 hoisted per gradient inside the NUTS
  machinery, 1 top-level; 0 in time loops. The dtrsm solves stay in the time
  loops in both arms, as they must (state-dependent).
- Analyzer: session-scratchpad analyze_potrf.py (classifies custom_calls by
  result signature `(DxD, i64)` = potrf vs `Dx1` = trsm, by leaf-while
  nesting).

## Mechanism note

The cholesky case is a sharper "post-lowering cannot recover" witness than
the matmul rows: after AD, the unhoisted loop contains a per-iteration
Cholesky pullback whose cotangent input varies with t, so no LICM at any
stage applies; only pre-AD hoisting collapses the adjoint to one post-loop
pullback (linearity of the pullback in the cotangent). Matmul adjoint chains
(Cubic Kernel) can be partially restructured by loop AD; the factorization
cannot.

## Provenance / honesty (paper framing)

- Problem class: standard Bayesian coefficient inversion for the heat
  equation (Stuart 2010 Acta Numerica framing; inverse-crime protocol per
  Kaipio & Somersalo 2005). The INSTANCE (grids, sensors, priors, scales) is
  constructed for this paper — present as "a representative Bayesian inverse
  problem of the standard class", not as an existing published benchmark.
- The in-loop factorization is intrinsic: the heat equation is stiff
  (explicit-Euler limit dt <= h^2/2k exceeded 25x at D=100, 227x at D=300),
  so implicit stepping is required; frozen-Jacobian factorization reuse is
  the hand-engineered optimization of production stiff solvers, recovered
  here automatically from the naive source form.
- Disclosure: M is tridiagonal; a sparse-solver expert would use an O(D)
  banded solve. Dense factorization is what tensor-based PPL stacks offer;
  the 1D rod stands in for 2D/3D operators where dense/blocked is realistic.
- SICM: the model class carries the EigenLift structure (M = I + dt*k*L is a
  scalar shift of a fixed operator), and was selected in July partly for it
  (no new rewrite was needed — CholeskyEigenLift matched as-is). Per user
  decision 2026-08-21, heat appears ONLY in the pre-lowering table; no SICM
  ablation entry.

## Repro

```
cd oopsla/Reactant.jl
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/heat_inverse.jl <A|B|R> <1|3> bench
# region + LICM off (Roff context cell):
HEATINV_DISABLE_LICM=1 ... heat_inverse.jl R <1|3> bench
```
