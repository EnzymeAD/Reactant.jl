# tab:prelowering: provenance-bearing substitution for the two synthetic rows (2026-08-25)

The Cubic Kernel (both sizes) and Projected Dynamics rows are retired as too
synthetic and replaced by two provenance-bearing benchmarks measured under the
same two-arm protocol. The Heat Equation rows stand unchanged.

Final figure rows: **Heat Equation**, **Heat Equation (large)**,
**Advection-Diffusion**, **STAR (ILINet)**.

- **Row A (in-loop factorization, second physics)**: IMEX advection-diffusion
  contaminant-transport inverse problem. Same mechanism as heat (a
  loop-invariant Cholesky kept inside the time loop in source), in a second
  physics, with a loop-VARIANT explicit advection flux in the same loop body,
  which makes it a stricter test of the LICM pattern than heat.
- **Row B (construction mechanism, real structure)**: STAR(1) space-time
  autoregression on a real US state contiguity graph and a real CDC ILINet
  series. This is the provenance-bearing instance of the role Projected
  Dynamics played, and its speedup is correspondingly modest.

## SUPERSEDING NOTE (frozen-adapted-state protocol)

**The figure numbers in this file's original "Table numbers" section are
superseded.** They were measured from a naive fixed step size of 0.02 starting
at the prior mean. Per the 2026-08-25 protocol revision the figure timings now
come from a FROZEN-ADAPTED-STATE regime, reported in the appended section
"Frozen-adapted-state measurement" at the end of this file. The earlier
adaptation-off-from-scratch batches remain VALID and are retained as
matched-trajectory validation runs; what changed is the regime the figure
timings are taken in, not the arm definition or any verification gate.

The revision matters because a fixed step of 0.02 is 25x to 38x smaller than
what warmup actually adapts to on these models, so the naive protocol measured
an unrealistically cheap regime.

Figure-facing arm names also changed on 2026-08-25: "Pre-Opt" is now
**Materialize with Standard Optimizations** and "No Pre-Opt" is now
**Materialize without Standard Optimizations**. This file keeps the short
historical names when describing past batches; only the figure display strings
and the wrapper caption changed. Internal JSON keys stay `preopt` / `noopt`.

## Protocol (unchanged from heat_inverse_RESULTS_2026-08-21.md)

20 NUTS samples, adaptation off, CPU (`CUDA_VISIBLE_DEVICES=""`), one warm-up
call then medians of 5 timed runs per cell, no thread caps, matched seeds,
per-cell `syevd = 0` checks. Each model keeps its own sampler settings: heat and
advection-diffusion use step 0.02, STAR uses step 0.02; all use
`max_tree_depth = 5`.

**Arms (NO SICM machinery on either side).**

- **A (No Pre-Opt)**: flat expand path (`DEBUG_PROBPROG_DISABLE_OPT=true`) +
  `disable_licm_optimization_passes=true`.
- **B (Pre-Opt)**: flat expand path + full standard passes.
- The ONLY cross-arm difference is the LICM pattern subset. The region/SICM
  pipeline is off in both arms, `DEBUG_PROBPROG_DISABLE_SICM=true` is set
  explicitly, and `syevd = 0` is verified in every measured cell.

**Binary**: `libReactantExtra.so` md5 `21a6bd08b37b54d2b8a20d8924a2bbb2`
(mtime 2026-08-24 14:59). Verified per cell by the batch driver, which also
aborts-and-flags if it ever changes. This is the same build the sibling SICM
campaign is using, so the two tables share a build.

**Single-build, two-batch (disclosure).** The Heat rows come from the quiet
2026-08-24 batch 2 (`logs_batch2/`, load1 2.5-3.4). The Advection-Diffusion and
STAR rows come from the 2026-08-25 batch (`logs/`). Both ran on the md5 above.
The heat rows were re-measured on 2026-08-25 as a cross-check and that
measurement was discarded as contaminated; see "Contamination exhibit" below.

Machine: AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, 1 NUMA
node. Julia 1.11.7.

## Table numbers

| Model | Size | Pre-Opt (B) | No Pre-Opt (A) | Speedup | Mechanism |
|---|---|---|---|---|---|
| Heat Equation | D=100, T=100 | 490 ms | 2936 ms | 6.0x | in-loop factorization |
| Heat Equation (large) | D=300, T=100 | 2794 ms | 18705 ms | 6.7x | in-loop factorization |
| Advection-Diffusion | D=200, T=100 | 2315 ms | 22871 ms | **9.9x** | in-loop factorization (IMEX) |
| STAR (ILINet) | N=47, T=104 | 29 ms | 204 ms | **7.0x** | in-loop construction |

Heat rows: 2026-08-24 batch 2, unchanged, already staged in the figure data.
Advection-Diffusion and STAR: 2026-08-25 batch (`logs/`).

Per-run timings (all 5 runs, ms):

| Cell | runs |
|---|---|
| Advection-Diffusion, Pre-Opt | 2618.8, 1509.4, 2173.1, 2352.3, 2314.6 |
| Advection-Diffusion, No Pre-Opt | 26552.9, 15187.3, 20928.2, 22871.0, 23994.9 |
| STAR, Pre-Opt | 24.8, 29.1, 32.3, 31.6, 25.2 |
| STAR, No Pre-Opt | 195.0, 204.4, 234.4, 234.7, 183.5 |

Compile times (excluded from the medians): 29.8 s / 31.0 s for the
advection-diffusion arms, 21.3 s / 20.5 s for the STAR arms.

Two notes on reading the table. Advection-Diffusion is the largest speedup in
the set, and it should be: at D=200 it eliminates both the per-step dpotrf and
the per-step O(D^3) Cholesky pullback from the reverse pass (10 of each per
gradient, see IR evidence), while heat at D=100 eliminates a smaller absolute
amount of the same work. STAR at 7.0x is higher than the 4.1x its retired
predecessor Projected Dynamics reported, because the STAR recurrence runs 103
steps against Projected Dynamics' 100 at a smaller N but eliminates a larger
fraction of the per-step work (the whole `rho*W + phi*I` assembly plus two
Frobenius adjoint contractions, against Projected Dynamics' single elementwise
construction). It remains the smallest-mechanism row of the four and the only
one whose invariant is not a factorization.

## Row A: Advection-Diffusion contaminant transport

### Model

    kappa = exp(z_kappa),  v = exp(z_v),  sigma = exp(z_sigma)
    M(kappa) = I + dt*kappa*L                       (SPD, loop-invariant)
    M(kappa) x_{t+1} = x_t - dt*v*(G x_t)           (IMEX step)
    y_j ~ Normal(H x_{t_j}, sigma^2 I)

L is the SPD negative Laplacian and G the fixed first-order upwind gradient
stencil. Diffusion is taken implicitly (it is the stiff term) and advection
explicitly, so each step is one Cholesky of M plus two triangular solves plus
one mat-vec. M does not depend on t, so the factorization is loop-invariant
while the right-hand side is not.

Three inferred parameters (log diffusivity, log velocity, log noise) against
S=8 sensors at M_obs=10 observation times. D=200, T=100, dt=0.0025.

### Stability bounds (both reported, as required)

| Quantity | Coarse (inference) grid | Fine (data) grid |
|---|---|---|
| dt / explicit-diffusion limit `h^2/(2 kappa)` | **20.2x** (implicit stepping required) | 40.4x |
| advection CFL `v*dt/h` | **0.754** (< 1, explicit half stable) | 0.754 |
| Peclet `v*L/kappa` | **15.0** (advection-dominated) | 15.0 |

The stiffness ratio is the analogue of heat's 25x at D=100, so the in-loop
factorization is intrinsic here for the same reason: explicit diffusion would
need a 20x smaller step. The CFL number sits comfortably below 1, so the
explicit half of the IMEX split is defensible rather than marginal.

### Identifiability (why the transport parameters are real)

The first configuration tried (kappa=0.5, v=0.5) was advection in name only:
the advective part of the plume displacement was ~9% of the total, and a 10%
change in v moved the sensor readings by 0.04 sigma, so v was effectively
unidentified. The shipped configuration (kappa=0.10, v=1.50) fixes this while
keeping both stability bounds:

| Perturbation | max abs sensor change | in sigma units | chi^2 shift over the 80 observations |
|---|---|---|---|
| kappa +10% | 0.0166 | 0.83 | 7.1 |
| v +10% | 0.0213 | 1.07 | 35.5 |

74% of the centre-of-mass displacement is advective, and sigma=0.02 is 4% of
the peak sensor reading. Posterior geometry at the prior mean is benign: the
log density is finite, and the curvature eigenvalues (0.58, 1.02, 4.20) imply a
stable leapfrog step near 0.98 against the 0.02 the harness uses, so trees
build to depth 5 and every sample does real work.

### Provenance framing (same rules as heat)

Problem class: standard Bayesian parameter identification for contaminant
transport, posed in the standard framework for Bayesian inverse problems
(Stuart 2010 Acta Numerica) with the inverse-crime protocol of Kaipio &
Somersalo (2005), data generated on a 2x finer grid at dt/2. The INSTANCE
(grid, sensors, priors, scales) is constructed for this paper and must be
presented as "a representative Bayesian inverse problem of the standard
class", never as an existing published benchmark.

### IR evidence (dump-first, D=200)

LAPACK custom_calls classified by result signature and by `stablehlo.while`
nesting depth. Depth 0 is top level (the InitHMC gradient), depth 1 is the
InitHMC gradient's own model time loop, depth 3 is per leapfrog step outside
the model time loop, depth 4 is the innermost model time loop.

| Arm | potrf d0 | potrf d1 | potrf d3 | potrf d4 | potrf total | trsm d1 / d4 |
|---|---|---|---|---|---|---|
| A (No Pre-Opt) | 1 | 10 | 1 | 10 | **22** | 60 / 60 |
| B (Pre-Opt) | 1 | 0 | 1 | 0 | **2** | 60 / 60 |

22 potrf in arm A with 20 of them inside the time loops; 2 in arm B with none
in the time loops. This reproduces heat's frozen structure exactly, in a second
physics. The dtrsm solves stay in the time loops in both arms, as they must.

The O(D^3) reverse-pass matmul chain moves with the factorization:

| Arm | O(D^3) dot_generals at d4 (time loop) | at d1 | at d3 | at d0 |
|---|---|---|---|---|
| A | **10** | 10 | 0 | 0 |
| B | **0** | 0 | 10 | 10 |

Crucially, the loop-VARIANT work correctly stays put: the `200x200 @ 200x1`
mat-vecs (the advection flux G x and the solve applications) remain at depth 4
in BOTH arms, 20 of them. So Pre-Opt hoists exactly the invariant factorization
and its adjoint, and nothing else. **No LICM coverage gap for this row**: the
pattern set handles the shape even though the loop body is no longer purely
"invariant plus a solve".

## Row B: STAR space-time autoregression on ILINet

### Data provenance decision: option (1), real adjacency AND real series

Procurement of option (1) succeeded, so the fallback to the in-tree NYC
250-tract map with simulated series was not needed.

- **W**: contiguity of the contiguous US states, derived deterministically from
  the US Census Bureau county adjacency file
  `https://www2.census.gov/geo/docs/reference/county_adjacency.txt` (two states
  are neighbours iff some county of one is listed adjacent to some county of
  the other). This reproduces the classic `spdep` usa48 contiguity graph from a
  primary source rather than shipping an R object. Row-standardised, which is
  the STAR/Cliff-Ord convention and bounds the spectral radius at 1.
- **Y**: CDC ILINet weekly state-level influenza-like-illness rates, field
  `ili`, retrieved through the Delphi Epidata mirror of CDC FluView
  `https://api.delphi.cmu.edu/epidata/fluview/`. Transformed as
  `log(ili + 0.1)` and centred per state.
- **Instance**: N=47, T=104 (epiweeks 201540 to 201739), 107 undirected edges,
  degree min/mean/max 1/4.55/8 (the degree-1 node is Maine, which is correct).
- **Exclusions, both principled and disclosed**: Florida is dropped because it
  does not report a state-level ILINet series at all (the API returns "no
  results"). Louisiana is dropped because its series starts at 201640 and so
  covers only 52 of the 104 weeks. The node-selection rule is "fix the
  observation window, then keep the states that cover it completely"; the
  alternative (fix the state set, truncate the window to the intersection)
  would have halved T from 104 to 52 for the sake of one late-starting state.
- Both W and Y are real observed data. Nothing is simulated.
- Build script: `build_star_ili.py`. Output:
  `pplbench/pplbench/models/real_data/star_ili_states.json`, with a
  `_provenance` block in the house style.

Procurement caveat for reproducers: the Delphi Epidata API rate-limits
anonymous queries (HTTP 429) after roughly one query per state. The build
script caches per-state responses under `/tmp/sbrantq-star`, so a rerun is
free, but a cold rebuild of a much longer window would need a free API key.
This is why the window is 104 weeks rather than the full 2010-2019 span.

### The formulation finding (a real negative result, reported not dodged)

The first formulation used the standard CONDITIONAL (one-step-ahead) STAR
Gaussian likelihood,

    y_{t+1} | y_t ~ Normal( K(rho, phi) y_t , sigma^2 I ),  K = rho*W + phi*I

which is the textbook estimator. **It has no loop-carried state**: each step is
independent given the data, so the loop over t is a map-reduce, and Reactant's
loop-raising rewrites the entire 103-step loop into a single batched
`103x47 @ 47x47` matmul. The consequence:

- `stablehlo.while` count is 4 in both arms (the NUTS machinery only; the model
  time loop is gone entirely), against 10 when the loop survives.
- Arm A and arm B IR are structurally identical. Diffing the two dumps shows
  differences only in opaque JIT `backend_config` pointer bytes and in the
  ordering of two `broadcast_in_dim`/`multiply` pairs. File sizes differ by 6
  bytes out of 192 kB.
- There is therefore nothing in a loop for LICM to hoist and the row would have
  been vacuously 1.0x.

**This is not a gap in the LICM pattern set.** It is the compiler doing
something strictly better than LICM (eliminating the loop rather than hoisting
out of it), and it is a property of the formulation. The mechanism this table
measures needs a loop-carried recurrence, not merely a loop.

The shipped row therefore uses the recurrence the row is supposed to be about,
which is also what the STAR model is defined as:

    x_1 = y_1 (the real week-1 field)
    x_{t+1} = K(rho, phi) x_t          (deterministic STAR propagation)
    y_t ~ Normal(x_t, sigma^2 I)

`x` is genuinely loop-carried, so the loop survives. This is the state-space
reading of the STAR recurrence and it is structurally identical to what the
heat and advection-diffusion rows do (deterministic dynamics from a
data-derived initial condition, observed with noise).

Both forms are kept in the harness behind `STAR_FORM=recur|cond` so the
negative result stays reproducible.

**Scale-parameter note (documented choice).** The conditional STAR likelihood
does not separately identify an innovation scale and an observation scale
without a latent-state filter, and adding a Kalman filter would replace the
in-loop operation with per-step covariance recursions, i.e. a different
mechanism from the one this row exists to measure. The row therefore samples
three parameters: the spatial autoregressive coefficient rho, the temporal
autoregressive coefficient phi, and one log scale.

### IR evidence (dump-first, N=47, recurrent form)

The invariant here is the O(N^2) elementwise construction `K = rho*W' + phi*I`,
not a factorization, so the evidence is the D x D construction ops by depth:

| Arm | depth 0 | depth 1 | depth 3 | depth 4 (time loop) |
|---|---|---|---|---|
| A (No Pre-Opt) | mul x2, bcast x2, add | mul x2, bcast x2, add, reshape | mul x2, bcast x2, add | **mul x2, bcast x2, add, reshape** |
| B (Pre-Opt) | mul x2, bcast x2, add | **add** | mul x2, bcast x2, add | **add** |

Read the table by the two time-loop columns. Arm A carries the full K
construction at all four depths, including both model time loops (depth 4 for
the sampling gradient and depth 1 for the InitHMC gradient). Arm B reduces both
of those to a single N x N add, which is the genuinely loop-carried `Kbar`
adjoint accumulation and must stay. The depth-0 and depth-3 copies are present
in both arms and are the out-of-loop positions the construction is hoisted to.
The two `47x47 @ 47x47 -> f64` Frobenius contractions (the rho and phi adjoint
accumulations) move from depth 4 to depth 3 in the same way. The loop-variant
`1x47 @ 47x47` mat-vec (x3, the recurrence and its reverse instances) correctly
stays at depth 4 in both arms.

This is the Projected Dynamics structure, on real data. As with Projected
Dynamics, the eliminated construction is O(N^2) rather than a factorization, so
the expected and observed speedup is modest. That is the point of the row: it is
the small-mechanism companion to the two factorization rows, and the contrast
between them is part of the claim.

## Verification gates

All gates pass on all four rows.

| Row | accept decisions identical | max abs sample diff | max abs log-density diff | syevd A / B | n_accepted / n_diverging |
|---|---|---|---|---|---|
| Heat Equation | yes | 7.4e-12 | n/a (parity mode) | 0 / 0 | n/a |
| Heat Equation (large) | yes | 8.7e-11 | n/a (parity mode) | 0 / 0 | n/a |
| Advection-Diffusion | yes (divergence flags too) | 2.6e-13 | 2.0e-11 | 0 / 0 | 20 / 0 |
| STAR (ILINet) | yes (divergence flags too) | 1.1e-16 | 0.0 | 0 / 0 | 20 / 0 |

Verification seeds: `[1, 5]` for the heat rows (`heat_inverse.jl <arm> <scale>
parity`, one JSON per arm), `[7, 11]` for the two new rows (folded into the
bench run, so one compile per cell).

STAR's log densities are bit-identical between arms and its samples differ by
one ULP, which is what an O(N^2) elementwise hoist should do. The
advection-diffusion row sits at the LAPACK/fusion reordering floor, like heat.
Both new rows accept all 20 proposals with zero divergences, so the chains are
healthy and the two arms are exploring identical trajectories.

The `syevd = 0` check matters most for the advection-diffusion row: like heat,
its `M = I + dt*kappa*L` is a scalar shift of a fixed operator and is therefore
exactly CholeskyEigenLift's target pattern, so SICM would fire if it were
enabled. It is off by construction (the flat expand path never runs the sicm
pass) and by explicit flag (`DEBUG_PROBPROG_DISABLE_SICM=true`), and verified
per cell by the absence of syevd.

## Machine conditions

The 2026-08-25 driver waits indefinitely for a quiet machine: it requires
`load1 < 8` AND no `pplbench`/`harness.jl` process before every cell, with no
proceed-anyway ceiling. It also records the binary md5 per cell and flags any
cell whose run overlapped a sibling process.

| Cell | load1 before | load1 after | sibling before / after | md5 |
|---|---|---|---|---|
| advdiff_B | 7.44 | 23.68 | 0 / 0 | unchanged |
| advdiff_A | 7.29 | 76.94 | 0 / 0 | unchanged |
| star_B | 7.66 | 7.02 | 0 / 0 | unchanged |
| star_A | 7.02 | 6.72 | 0 / 0 | unchanged |

Every `load1 after` above the gate is self-inflicted: the advection-diffusion
arms are the threaded-LAPACK-heavy cells, and the 1-minute average tracks this
process's own thread count. The `sibling_after = 0` column is the important
one: no cell overlapped the sibling SICM campaign.

The gate waited 2130 s (35.5 min) before the first cell, while the sibling
campaign's `sar_boston` job held 96 cores at load1 130 to 200, then 270 s
before each of the next two cells while this process's own load decayed. The
whole batch, gate waits included, ran 13:26:42 to 14:17:24.

Binary md5 `21a6bd08b37b54d2b8a20d8924a2bbb2` verified at batch start, after
every cell, and at batch end. Unchanged throughout, so this table and the SICM
table share a build.

For the Heat rows the corresponding conditions are those of 2026-08-24 batch 2,
recorded in `logs_batch2/batch_meta.txt`: external load1 2.5 to 3.4 across the
cubic/projected cells and 3.0 at the heat_s1_B gate, with the later rise during
the heat cells self-inflicted for the same LAPACK reason.

### Contamination exhibit: why the load gate exists

The 2026-08-25 heat re-measurement was discarded. Its `heat_s3_A` cell ran
13:07 to 13:10 while an unrelated 302% `julia` process (another user) was
active, and the sibling SICM campaign's `sar_boston` job (`--threads all`, 96
cores, load1 peaking above 200) started minutes later.

| Cell | Contended (2026-08-25, discarded) | Quiet (2026-08-24, batches 1/2/3) |
|---|---|---|
| heat_s3_B | 2883.3 ms | 2695.0 / 2794.0 / 2764.9 ms |
| heat_s3_A | **23444.9 ms** (runs 18775, 17278, 27968, 23445, 27171) | 18635.3 / 18705.4 / 18559.5 ms |
| implied speedup | **8.13x** | 6.74 / 6.69 / 6.89x |

Arm B moved about 4% while arm A moved about 26%, because arm A is the
threaded-LAPACK-heavy side and therefore the side that loses most when cores
are contended. The result is a speedup inflated by roughly 1.2x purely from
machine load. A reader would have had no way to tell from the number alone.
This is preserved in `logs_v2batch1_CONTENDED/` and is the concrete reason the
2026-08-25 driver waits indefinitely for a quiet machine rather than proceeding
after a timeout.

## Does the heat scaling pair earn its second row?

Evidence from the three quiet 2026-08-24 batches (same binary):

| Size | speedup per batch | median | range |
|---|---|---|---|
| D=100, T=100 | 6.65 / 5.99 / 6.19 | 6.19 | 5.99 to 6.65 |
| D=300, T=100 | 6.74 / 6.69 / 6.89 | 6.74 | 6.69 to 6.89 |

- The ordering is consistent: D=300 beats D=100 in 3 batches out of 3, and the
  two ranges do not overlap (6.65 against 6.69).
- But the growth is small: median 6.19 to 6.74, i.e. **1.09x**, against a
  flop-bound prediction of `T*D/(D+T)` growing 50 to 75, i.e. **1.50x**. So the
  pair does NOT demonstrate the predicted scaling; it demonstrates that the
  speedup is roughly flat in D. The compression cause is the one already
  documented for heat: the eliminated dpotrf is threaded BLAS-3 while the
  remaining dtrsm solves are memory-bound BLAS-2.
- What the second row does buy is a 5.8x span in absolute Pre-Opt cost (480 ms
  to 2765 ms) with the mechanism intact.

**Recommendation (user decides).** The second heat row is weakly informative as
a *scaling* claim and should not be narrated as one. If a compact figure is
wanted, Heat (large) is the row to drop, because Advection-Diffusion at D=200
now supplies an independent second factorization point at a third problem size
and in a different physics, which covers the "the mechanism holds as models
grow" reading more convincingly than a same-model size pair. If the figure
keeps four rows, Heat (large) is harmless provided no scaling language attaches
to it.

## Retired rows (superseded, preserved, not deleted)

Measured 2026-08-24 under the identical two-arm protocol and the same binary,
headline batch 2 of 3 agreeing batches. Full detail in
`cubic_projected_RESULTS_2026-08-24.md` and `cubic_projected_RAW_2026-08-24.json`.

| Model | Size | Pre-Opt | No Pre-Opt | Speedup | 3-batch band |
|---|---|---|---|---|---|
| Cubic Kernel | D=60, T=50 | 57 ms | 1172 ms | 20.7x | 19.3 to 22.0x |
| Cubic Kernel (large) | D=100, T=100 | 170 ms | 6174 ms | 36.3x | 32.0 to 36.3x |
| Projected Dynamics | D=100, T=100 | 35 ms | 143 ms | 4.1x | 4.0 to 4.2x |

These are retired for being synthetic (`randn` weight matrices, no data
provenance), not for being wrong. Advection-Diffusion inherits the
factorization role from the Cubic Kernel rows and STAR inherits the
construction role from Projected Dynamics.

## Findings and coverage gaps

1. **No LICM coverage gap was found for either new row.** Both the
   advection-diffusion Cholesky (with a loop-variant flux in the same body) and
   the STAR elementwise construction are hoisted correctly by the Pre-Opt arm,
   and the loop-variant work stays in the loop in both arms.
2. **Loop-raising subsumes LICM for map-reduce loops** (the STAR conditional
   form above). Any future pre-lowering row must be checked for a genuine
   loop-carried dependence before it is measured, or it will silently read
   1.0x. This is a good compiler outcome, but it narrows the set of models for
   which the pre-lowering LICM claim is the operative one, and that is worth a
   sentence in the paper if the reviewers push on generality.
3. **Loop-carry typing**: a `@trace` loop carry must START as a traced value.
   `zeros(1, N)` fixes the carry type to `Matrix{Float64}` and the traced body
   value then fails to convert back (scalar-indexing error), and
   `reshape(Y[1, :], 1, N)` produces a lazy `Base.ReshapedArray` wrapper which
   fails the same way. `Y[1:1, :] .+ 0.0` works. This cost two compile cycles
   and is worth knowing for anyone writing a new traced-loop model.
4. **Smoke scales can hide the mechanism**: at scale 0 the advection-diffusion
   time loops have trip count 2 and are unrolled, so both arms produce
   identical IR and the dump-first check is vacuous. Dump-first must be run at
   the measured scale.
5. **refs.bib gap** (paper-side, not actioned here since I do not edit the
   paper): the STAR row needs Pfeifer & Deutsch (1980) and a Cliff-Ord entry;
   `refs.bib` currently has `stuart2010inverse` and `kaipio2005statistical`
   (both reusable for the advection-diffusion row) and `cressie1993spatial`,
   but no STAR-specific citation.

## Harness changes

New files (all additive; nothing existing was restructured):

- `advection_diffusion_data.jl`, `advection_diffusion.jl`: Row A. Data module
  mirrors `heat_inverse_data.jl` (scale presets, inverse-crime fine-grid data
  generation, JSON dump) and adds the stability diagnostics. Harness mirrors
  `heat_inverse.jl` arm-for-arm and uses `prelowering_common.jl`.
- `star_ili.jl`, `build_star_ili.py`: Row B and its data builder.
- `run_prelowering_batch_v3.sh`: the 2026-08-25 driver. Same protocol as
  `run_prelowering_batch.sh` plus (a) per-cell binary md5 check with an explicit
  BINARY CHANGED flag, (b) a sibling-campaign detector that flags a cell whose
  run overlapped a `pplbench`/`harness.jl` process, and (c) a load gate with NO
  proceed-anyway ceiling.
- `run_prelowering_batch_v2.sh`: superseded by v3 (it had a 60-minute
  proceed-anyway ceiling). Kept for the record.
- `aggregate_v2.py`: aggregation for the new row set.

Unchanged: `heat_inverse.jl`, `heat_inverse_data.jl`, `heat_inverse_v2.jl`,
`prelowering_common.jl`, `analyze_ir.py`, `run_prelowering_batch.sh`,
`cubic_kernel_dynamics.jl`, `projected_dynamics.jl`.

## Files

- `advdiff_star_RAW_2026-08-25.json`: all 5 per-run timings per cell, per-cell
  IR counts, verification samples and accept decisions, compile times, batch
  metadata.
- `logs/`: the 2026-08-25 batch (Advection-Diffusion, STAR).
- `logs_batch2/`: the 2026-08-24 batch supplying the Heat rows.
- `logs_v2batch1_CONTENDED/`: the discarded contended heat re-measurement.
- `dumpfirst/`: dump-first IR dumps and logs, including
  `star_cond_arm{A,B}.mlir` (the loop-raising negative result) and
  `star_recur_arm{A,B}.mlir`.

## Repro

```
cd oopsla/Reactant.jl
python3 benchmark/probprog/ablation/pre_lowering/build_star_ili.py   # once
bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch_v3.sh
```

Per cell:

```
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/advection_diffusion.jl <A|B> 1 <bench|ir>
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/star_ili.jl <A|B> <bench|ir>
STAR_FORM=cond ... star_ili.jl <A|B> ir      # the loop-raising negative result
```

IR evidence:

```
cd oopsla/Reactant.jl/benchmark/probprog/ablation/pre_lowering
python3 analyze_ir.py potrf 200 logs/hlo/advdiff_scale1_arm{A,B}.mlir
python3 analyze_ir.py dots  200 logs/hlo/advdiff_scale1_arm{A,B}.mlir
python3 analyze_ir.py elt   47  logs/hlo/star_recur_arm{A,B}.mlir
python3 analyze_ir.py dots  47  dumpfirst/star_cond_arm{A,B}.mlir
```

---

# Frozen-adapted-state measurement (2026-08-25, supersedes the figure numbers above)

## Protocol

Per model, in two phases:

1. **Warmup (once per model, NOT timed).** Run with adaptation ON (dual-averaging
   step size + Welford inverse mass) for 200 iterations in the
   materialize-with-standard-optimizations arm. Save the adapted step size, the
   adapted inverse mass matrix and the post-warmup position to
   `adapted_states/<model>.json` with provenance.
2. **Measured (both arms).** Adaptation OFF, initialised from that SAME saved
   state: identical step size, identical inverse mass, identical starting
   position, matched seeds. The arms therefore walk bit-identical trajectories
   and the identical-accepts gate remains the arms-equivalence check, while the
   timings are taken in the realistic adapted regime.

20 NUTS samples per execution, median of 5 executions as headline, geomean of
the same executions also reported. No compiler changes were needed: the
`mcmc_logpdf` entry point already takes `initial_position` positionally and
`step_size` / `inverse_mass_matrix` as keywords, and already returns the
post-run `MCMCState` carrying all three.

## Why the regime change matters

Warmup adapts to a step size 21x to 38x larger than the fixed 0.02 the earlier
batches used:

| Model | initial step | adapted step | ratio |
|---|---|---|---|
| Heat Equation (D=100) | 0.02 | 0.7666 | 38x |
| Heat Equation (D=300) | 0.02 | 0.6428 | 32x |
| Advection-Diffusion | 0.02 | 0.7406 | 37x |
| STAR (ILINet) | 0.02 | 0.4256 | 21x |

The naive protocol was therefore measuring a regime no real user would sample
in. All four adapted inverse mass matrices came back dense (rank 2), matching
the dense identity they were seeded with, per the rank-determines-diagonal
convention.

The advection-diffusion warmup is also a model check: it converged to
`z = (-2.331, 0.403, -3.936)` against the data-generating
`z = (log 0.10, log 1.50, log 0.02) = (-2.303, 0.405, -3.912)`, so the sampler
recovers all three transport parameters.

## Table numbers (frozen-adapted, v5 batch)

| Model | Size | Materialize with (B) | Materialize without (A) | Speedup (median) | Speedup (geomean) | median vs geomean |
|---|---|---|---|---|---|---|
| Heat Equation | D=100, T=100 | 320 ms | 1939 ms | **6.06x** | 6.05x | 0.1% |
| Heat Equation (large) | D=300, T=100 | 2652 ms | 17077 ms | **6.44x** | 6.50x | 1.0% |
| Advection-Diffusion | D=200, T=100 | 1277 ms | 12615 ms | **9.88x** | 10.01x | 1.3% |
| STAR (ILINet) | N=47, T=104 | 20 ms | 148 ms | **7.32x** | 7.46x | 1.9% |

Median and geomean agree within 2% on every row, so no divergence flag is
raised. Per-run timings (all 5 executions, ms):

| Cell | runs |
|---|---|
| Heat, with | 375.1, 293.9, 319.8, 386.4, 315.8 |
| Heat, without | 2205.4, 1774.7, 1938.9, 2379.1, 1938.7 |
| Heat large, with | 2824.5, 2651.6, 2270.0, 3307.7, 2559.9 |
| Heat large, without | 16776.6, 17077.2, 15180.6, 22190.5, 17328.6 |
| Advection-Diffusion, with | 1469.3, 1105.8, 1016.7, 1491.3, 1277.3 |
| Advection-Diffusion, without | 14301.5, 12153.2, 12614.9, 15146.2, 9501.3 |
| STAR, with | 20.6, 20.1, 24.6, 20.2, 16.2 |
| STAR, without | 147.6, 144.4, 174.1, 158.8, 130.2 |

Comparison with the superseded naive fixed-step numbers (6.0x / 6.7x / 9.9x /
7.0x): the speedups are essentially unchanged, which is the reassuring outcome.
The absolute times drop substantially in the adapted regime (heat D=100 goes
490 ms to 320 ms in the with-optimizations arm) because the larger step builds
shallower trees, but the ratio the figure reports is robust to the regime.

## Compile time

Compilation is excluded from every timing above. Geomean across the four
benchmarks: **27.4 s** with standard optimizations, **26.3 s** without. The two
arms are within 4% of each other, so a single figure can be quoted. Per
benchmark (heat D=100, heat D=300, advection-diffusion, STAR): 26.4 / 33.3 /
29.6 / 21.8 s with, and 25.6 / 31.4 / 29.7 / 20.0 s without.

## State-injection verification

Run before any measured batch on heat scale 0 (D=20, T=10, 2 parameters) by
`verify_state_injection.jl`; full output in
`logs/state_injection_verification_scale0.json`. All four checks pass.

**1. Injection takes effect (positive controls).** Each perturbation of the
injected state changes the trajectory, so none of the three pieces is being
silently ignored:

| Perturbation | max abs sample change |
|---|---|
| position + 0.05 | 0.428 |
| step size x 1.5 | 0.458 |
| inverse mass x 2 | 0.450 |
| inverse mass replaced by identity | 0.338 |
| inverse mass replaced by its own diagonal | 0.099 |

The identity row is the specific control that the injected matrix is not being
silently replaced by the default. The diagonal row is the control that the
off-diagonal entries are actually consumed, which is what would catch a silent
flattening; the adapted matrix has off-diagonal magnitude 1.16e-3 against a
diagonal of 4.42e-4 and 6.18e-2, so it is genuinely non-diagonal.

**2. Shape and convention round-trip.** The JSON round-trip is bit-exact for
step size, inverse mass and position, and preserves rank 2. Under the
rank-determines-diagonal convention a dense `diag(a,b)` and a rank-1 `[a,b]`
denote the same metric, and they produce identical trajectories: max abs sample
difference 0.0 with identical acceptance decisions. That equivalence exercises
the dense path end to end, including the momentum sampling that uses the mass
matrix rather than its inverse, so an ordering or flattening error would break
it. A transpose is not tested because the adapted inverse mass is a symmetric
covariance estimate, making a transpose unobservable and harmless; that is
stated rather than claimed as a test.

**3. Arms-equivalence (the gate).** From the same injected state with matched
seeds the two arms are **bit-identical** at smoke scale: identical acceptance
decisions, identical divergence flags, max abs sample difference 0.0, max abs
log-density difference 0.0.

**4. Continuation equivalence (regime, not bit-identity).** A
warmup-then-restore run sits in the same regime as the post-warmup phase of a
single continuous adaptation-on run:

| Quantity | continuous run | restored run |
|---|---|---|
| step size | 0.4582835247608289 | 0.4582835247608289 (exact) |
| acceptance rate | 1.0 | 1.0 |
| per-sample cost | 0.186 ms | 0.179 ms (4% apart) |
| mean log density | 89.07 | 89.78 |
| posterior mean | (-0.663, -3.849) | (-0.664, -3.869) |
| posterior sd | (0.020, 0.108) | (0.013, 0.100) |

Per-sample cost is the observable proxy for tree depth, which the diagnostics
array does not expose. The continuous run's raw wall time cannot be divided by
the sample count directly: it is dominated by its 200 warmup iterations, which
run at the tiny initial step and therefore build far deeper trees. The figure
above subtracts a (warmup + 1 sample) run to isolate the continuous run's
post-warmup per-sample cost.

## Verification gates on the measured batch

| Row | accepts identical | divergences identical | max abs sample diff | syevd A / B |
|---|---|---|---|---|
| Heat Equation | yes | yes | 2.7e-13 | 0 / 0 |
| Heat Equation (large) | yes | yes | 1.8e-12 | 0 / 0 |
| Advection-Diffusion | yes | yes | 2.8e-13 | 0 / 0 |
| STAR (ILINet) | yes | yes | 1.8e-16 | 0 / 0 |

All rows now carry the gate under the exact measured configuration. Heat
previously used a separately-configured `parity` mode for this; its `bench`
mode now emits the verification run itself, so the gate covers what was timed.

## Machine discipline and the v4 contamination incident

The first frozen-adapted batch (v4, preserved in `logs_v4_frozen_contaminated/`)
produced a physically impossible Advection-Diffusion row: **0.61x**, i.e. the
with-optimizations arm apparently slower than the without arm, even though that
arm strictly removes work per gradient. The arms were verified
trajectory-identical at three separate timed seeds (101, 103, 105: identical
accepts, identical divergences, samples agreeing to ~1e-13), so equal work was
being done and the timing had to be wrong.

The cause was machine contamination that the v3/v4 gate could not see. That
gate matched only `pplbench|harness.jl` and sampled load only before and after
each cell. During `advdiff_B` an unrelated external job drove load1 to **240**,
and the cell's five executions came out bimodal (5335, 5875, 22325, 27507,
20682 ms). Load audit of the whole v4 batch:

| Cell | load before | load after | verdict |
|---|---|---|---|
| heat_s1_B / heat_s1_A | 7.20 / 6.56 | 6.56 / 17.63 | clean |
| heat_s3_B / heat_s3_A | 7.29 / 7.40 | 22.09 / 63.14 | suspect |
| advdiff_B / advdiff_A | 7.34 / 7.97 | **240.11** / 61.79 | contaminated |
| star_B / star_A | 7.11 / 6.89 | 6.89 / 6.59 | clean |

Raw load is a poor test on its own, because the heavy cells legitimately drive
it to ~70 through their own threaded LAPACK. The v5 driver therefore samples,
every 5 s for the duration of every cell, the summed CPU of every process NOT
owned by this user, records the maximum, and marks the cell contaminated and
retries it if that maximum exceeds 1000% (the baseline external load on this
shared machine is 400 to 550%, or 4 to 5 cores). Re-measured under v5,
`advdiff_B` came out at **1277 ms** against v4's 20682 ms, a **16x** measurement
error, and the row returned to 9.88x. The v5 batch fired zero contamination
flags on any cell and the binary md5 was verified per cell.

This is the second contamination incident in this campaign; the first, in the
2026-08-25 heat re-measurement, is recorded earlier in this file. Both had the
same shape: an external job arriving mid-cell and inflating the
threaded-LAPACK-heavy arm. The external-CPU monitor is the durable fix.

## Files added by this phase

- `verify_state_injection.jl`: the four-check verification harness.
- `run_prelowering_batch_v4.sh`: warmup + measured driver (superseded by v5 for
  the contamination detector, but it produced the adapted states in use).
- `run_prelowering_batch_v5.sh`: measured-phase driver with external-CPU
  monitoring, automatic retry, and selectable cells via `PRELOWER_CELLS`.
- `adapted_states/*.json`: the frozen sampler states, one per model, with
  provenance (producing arm, warmup seed, warmup length, initial step size).
- `logs/`: the v5 batch. `logs/batch_meta_v5_advdiff_star.txt` holds the meta
  for the advdiff and star cells, whose records predate the heat re-run that
  rewrote `batch_meta.txt`.
- `logs_v3_naive_fixedstep/`, `logs_v4_frozen_contaminated/`: superseded
  batches, retained.

## Harness changes for this phase

- `prelowering_common.jl`: added `extract_adapted_state`, `save_adapted_state`,
  `load_adapted_state` and the JSON row/matrix helpers.
- `heat_inverse.jl`, `advection_diffusion.jl`, `star_ili.jl`: added a `warmup`
  mode, adapted-state loading in `bench`, a geomean line, and a
  `PRELOWER_VERIFY_SEED` override. `heat_inverse.jl` additionally gained the
  cross-arm verification dump in `bench` and now shares the IR helpers from
  `prelowering_common.jl` rather than carrying duplicate copies.
- `aggregate_v2.py`: geomean, median-vs-geomean divergence, compile-time
  geomean across benchmarks, and adapted-state provenance per row.

## Note for the paper (NOT actioned here)

`figures/prelowering.tex` still describes the measurement as taken "with
adaptation disabled". Under the frozen-adapted protocol that is literally true
of the measured phase but incomplete, since the sampler is initialised from an
adapted step size and mass matrix rather than a naive fixed step. The arm
rename was applied to that caption as instructed; this phrase was left alone
because it is a substantive prose claim rather than a rename. It should be
updated by whoever owns the paper prose.
