# tab:prelowering under one uniform protocol (2026-08-24)

Re-measurement of the Cubic Kernel and Projected Dynamics rows under the exact
two-arm configuration frozen for the heat rows in
`heat_inverse_RESULTS_2026-08-21.md`, plus a re-measurement of the two heat
rows on the current binary, so that every cell of `tab:prelowering` comes from
one protocol and one build.

20 NUTS samples, adaptation off, CPU (`CUDA_VISIBLE_DEVICES=""`), one warm-up
call then medians of 5 timed runs per cell, no thread caps (uncapped ablation
regime), all cells of the reported table from ONE controlled batch. Each model
keeps its own sampler settings: heat uses step 0.02 / max_tree_depth 5; cubic
and projected use step 0.1 / max_tree_depth 5, as their harnesses already did.

**Binary: `oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so`,
md5 `21a6bd08b37b54d2b8a20d8924a2bbb2`, mtime 2026-08-24 14:59.** This is NOT
the binary the 2026-08-21 heat rows were frozen on (`b319ca908cc9b91dfbd734769ba1e700`,
mtime 2026-08-21 01:29); a sibling session rebuilt it. Per the single-build
rule the two heat cells were therefore re-measured in the same batches. The md5
was checked at the start and end of every batch and never changed.

Machine: AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, one NUMA
node. Julia 1.11.7.

## Arms (NO SICM machinery on either side)

- **A (No Pre-Opt)**: flat expand path (`DEBUG_PROBPROG_DISABLE_OPT=true`) +
  `disable_licm_optimization_passes=true` (strips the `*_licm` pattern subset
  from the standard passes; the option feeds both `opt_passes` before
  inference lowering and `opt_passes2` after AD, verified in
  `src/compiler/Compiler.jl` and `src/compiler/OptimizationPasses.jl:88`).
- **B (Pre-Opt)**: flat expand path + full standard passes.
- The ONLY cross-arm difference is the LICM pattern subset. The region/SICM
  pipeline is off in both. `DEBUG_PROBPROG_DISABLE_SICM=true` is set explicitly
  in both arms of all five models, and the flat path structurally never runs
  the `sicm` pass. Per-cell check: `syevd` (EigenLift's signature op) = 0 in
  the final IR of all 30 measured cells (5 models x 2 arms x 3 batches).
- The old Cubic Kernel / Projected Dynamics arm definition
  (`optimize=:probprog_no_opt` vs `:probprog`) is gone: `:probprog_no_opt` no
  longer exists in Reactant, and that axis was flat-vs-region, not
  pre-lowering LICM. Those two harnesses were edited to the arm mechanism above.

## Table numbers (headline: batch 2, start 21:17:39)

| Model | Size | Pre-Opt (B) | No Pre-Opt (A) | Speedup |
|---|---|---|---|---|
| Heat Equation | D=100, T=100 | 490 ms | 2936 ms | 6.0x |
| Heat Equation (large) | D=300, T=100 | 2794 ms | 18705 ms | 6.7x |
| Cubic Kernel | D=60, T=50 | 57 ms | 1172 ms | 20.7x |
| Cubic Kernel (large) | D=100, T=100 | 170 ms | 6174 ms | 36.3x |
| Projected Dynamics | D=100, T=100 | 35 ms | 143 ms | 4.1x |

Batch 2 was chosen as the headline because it ran under the lowest recorded
external load and because its speedup is at or nearest the three-batch median
in every row. It is a complete batch: no cell is imported from another run.

## Change from the previous table values

| Model | Old | New (batch 2) | Cause |
|---|---|---|---|
| Heat Equation | 6.9x | 6.0x | binary rebuild only (protocol unchanged) |
| Heat Equation (large) | 7.9x | 6.7x | binary rebuild only (protocol unchanged) |
| Cubic Kernel | 21x | 20.7x | new arm definition; lands on the old value |
| Cubic Kernel (large) | 36x | 36.3x | new arm definition; lands on the old value |
| Projected Dynamics | 7.7x | 4.1x | new arm definition |

The heat movement is a binary effect, not a protocol effect: arm B is
essentially unchanged (489 -> 490 ms at D=100; 2748 -> 2794 ms at D=300) while
arm A got about 13% faster on the new build (3392 -> 2936 ms; 21674 -> 18705
ms), which compresses the ratio. All three batches agree on this, so it is not
run-to-run noise.

Projected Dynamics is the row the protocol unification actually moves. Its old
7.7x came from the flat-vs-region axis, which bundled more than the LICM
pattern subset. Under the LICM-only definition the row reads 4.1x, which is
consistent with its mechanism: its loop-invariant construction is the O(D^2)
elementwise `K = theta*A + I`, not an O(D^3) factorization or matmul (see IR
evidence below), so it has the least to hoist and remains the smallest row.

## Run-to-run stability (three independent batches, same build)

| Model | Size | Pre-Opt min/med/max (ms) | No Pre-Opt min/med/max (ms) | Speedup min/med/max |
|---|---|---|---|---|
| Heat Equation | D=100, T=100 | 451.9 / 480.1 / 490.0 | 2935.5 / 2970.3 / 3005.1 | 6.0 / 6.2 / 6.6 |
| Heat Equation (large) | D=300, T=100 | 2695.0 / 2764.9 / 2794.0 | 18559.5 / 18635.3 / 18705.4 | 6.7 / 6.7 / 6.9 |
| Cubic Kernel | D=60, T=50 | 54.6 / 56.7 / 58.8 | 1137.2 / 1171.6 / 1200.7 | 19.3 / 20.7 / 22.0 |
| Cubic Kernel (large) | D=100, T=100 | 162.3 / 170.0 / 170.2 | 5203.2 / 6097.2 / 6174.0 | 32.0 / 35.9 / 36.3 |
| Projected Dynamics | D=100, T=100 | 34.3 / 34.8 / 35.2 | 140.6 / 143.4 / 145.1 | 4.0 / 4.1 / 4.2 |

Within-cell min-to-max spread over the 5 timed runs is 22-32% for the heat and
cubic-small cells and 82-119% for the cubic-large and projected cells. That
spread is **not** measurement noise: it is a reproducible warm-up ramp. Runs 1
through 5 trace nearly the same trajectory in every batch, e.g. Cubic Kernel
(large), arm B:

```
batch 1: [103.3, 147.8, 162.3, 232.6, 250.0] ms
batch 2: [107.8, 155.7, 170.2, 243.6, 254.7] ms
batch 3: [105.2, 153.9, 170.0, 242.8, 257.6] ms
```

A 25-run diagnostic on the noisiest cell (`jitter_diag/jitter_cubic_large_*.log`)
shows the ramp settling: arm B median 200.1 ms, arm A median 6207.9 ms over 25
runs, i.e. a steady-state ratio of 31.0x against the 32.0-36.3x band from the
medians of 5. A 25-run check of Projected Dynamics arm B gives a median of
32.8 ms against 34.3-35.2 ms from the medians of 5. Absolute medians of 5
therefore sit on the boost-clock part of the curve and read a little fast, in
both arms, and the ratios are stable to within the batch-to-batch band above.
No cell was tuned or re-run to reach a target value.

## Correctness / verification gates

All gates pass in all three batches.

- **Accept decisions**: with a matched seed, arms A and B make bit-identical
  accept decisions in every cell. Cubic and projected also agree on the
  divergence flags. Verification seeds: `[7, 11]` for cubic/projected (folded
  into the bench run of `cubic_kernel_dynamics.jl` / `projected_dynamics.jl`);
  `[1, 5]` for heat (`heat_inverse.jl <arm> <scale> parity`, one JSON per arm
  under `logs_batch*/parity/`).
- **Sample agreement** (max |A - B| over all 20 samples):

  | Model | max abs sample diff | max abs log-density diff |
  |---|---|---|
  | Heat Equation (D=100) | 7.4e-12 | n/a (parity mode) |
  | Heat Equation (D=300) | 8.7e-11 | n/a (parity mode) |
  | Cubic Kernel (D=60) | 2.4e-16 | 0.0 |
  | Cubic Kernel (large) | 1.1e-16 | 2.8e-17 |
  | Projected Dynamics | 2.6e-15 | 4.1e-15 |

  The cubic and projected rows are at the floating-point ULP floor (effectively
  bit-identical). The heat rows sit at the LAPACK/fusion reordering floor; the
  D=100 figure reproduces the frozen 7.4e-12 exactly. The D=300 figure, 8.7e-11,
  is above the ~1e-11 guideline in absolute terms; it is the same ULP floor
  scaled by the larger problem (samples are O(1), 20 sample values, 300x300
  factorizations per step), and it is identical to the digit across all three
  batches, so it is deterministic reordering rather than instability.
- **syevd = 0** in the final IR of every measured cell, both arms, all three
  batches (SICM-off proof; heat is the model that would otherwise match
  CholeskyEigenLift).

## IR evidence (final IR, classified by `stablehlo.while` nesting depth)

`analyze_ir.py` (shipped next to this file) classifies ops by how many
while-body regions enclose them. For these harnesses max nesting is 4:
depth 0 = top level (the InitHMC gradient), depth 1 = the NUTS sampling while
(and the InitHMC gradient's own model time loop), depth 3 = per leapfrog step
outside the model time loop, depth 4 = the innermost model time loop.

**Cubic Kernel (both sizes).** O(D^3) `stablehlo.dot_general` sites, FLOP-classified
(FLOPs = numel(result) x prod(LHS contracting dims); shapes alone are not
enough, since `(DxD, DxD) -> f64` is a Frobenius inner product, not a matmul):

| Arm | depth 0 | depth 1 | depth 3 | depth 4 |
|---|---|---|---|---|
| A (No Pre-Opt) | 1 | 3 | 1 | **3** |
| B (Pre-Opt) | 3 | 0 | 3 | **0** |

Arm A rebuilds `K = (theta W)^T (theta W) + I` inside the innermost time loop
every step, and the reverse pass adds two more per-iteration O(D^3) matmuls
there. Arm B has zero O(D^3) matmuls at either time-loop depth: depth 4 goes
3 -> 0 for the sampling gradient and depth 1 goes 3 -> 0 for the InitHMC
gradient, with the surviving sites sitting one while-level further out (3 at
depth 3 and 3 at depth 0). The per-region total falls from 4 to 3 on each side,
so the whole-program O(D^3) count drops from 8 to 6. The time loop keeps only
the O(D^2) `1xD @ DxD` state updates in both arms. The structure is identical
at D=60 and D=100. This is the matmul analogue of the heat rows'
20-potrf-to-0.

**Projected Dynamics.** No O(D^3) `dot_general` exists in either arm, which is
the row's whole point: its invariant `K = theta*A + I` is an O(D^2) elementwise
construction. Counting D x D elementwise/shape ops by depth:

| Arm | depth 0 | depth 1 | depth 3 | depth 4 |
|---|---|---|---|---|
| A (No Pre-Opt) | multiply, add, broadcast | multiply, add, broadcast | multiply, add, broadcast | **multiply, add, broadcast** |
| B (Pre-Opt) | multiply, add, broadcast | add | multiply, add, broadcast | **add** |

Arm B leaves only the loop-carried `Kbar` accumulation (one DxD add) in the time
loop; the construction itself is hoisted. The 30x30 `B'B` product of the
projection `P` is already outside the time loop in source and stays at depth
0/3 in both arms. The table lists the construction triple only; the shipped
`analyze_ir.py elt` output also reports the DxD `reshape`/`transpose` shape ops,
which follow the same depth-4 to depth-3 move (2 reshapes leave the time loop
in arm B).

**Heat Equation (D=100).** LAPACK custom_calls classified by result signature
(`(DxD, i64)` = dpotrf, `Dx1` = dtrsm):

| Arm | potrf depth 0 | potrf depth 1 | potrf depth 3 | potrf depth 4 | potrf total | trsm depth 1 / depth 4 |
|---|---|---|---|---|---|---|
| A (No Pre-Opt) | 1 | 10 | 1 | 10 | **22** | 60 / 60 |
| B (Pre-Opt) | 1 | 0 | 1 | 0 | **2** | 60 / 60 |

22 potrf in arm A, 20 of them inside the time loops; 2 in arm B, none in the
time loops. The dtrsm solves stay in the time loops in both arms, as they must
(state-dependent). This reproduces the frozen 2026-08-21 evidence exactly on
the new binary, which is the reason to trust the heat re-measurement as a
like-for-like refresh rather than a different experiment.

Repro of the evidence:

```
cd oopsla/Reactant.jl/benchmark/probprog/ablation/pre_lowering
python3 analyze_ir.py dots  60 logs_batch2/hlo/cubic_small_arm{A,B}.mlir
python3 analyze_ir.py dots  100 logs_batch2/hlo/cubic_large_arm{A,B}.mlir
python3 analyze_ir.py elt   100 logs_batch2/hlo/projected_arm{A,B}.mlir
python3 analyze_ir.py potrf 100 logs_batch3/hlo/heat_hlo_arm{A,B}_scale1.mlir
```

## Machine conditions

Load (1-minute average, 192 threads) was recorded before and after every cell
in `logs_batch*/batch_meta.txt`. For the headline batch 2:

| Cell | load1 before | load1 after |
|---|---|---|
| cubic_small B / A | 2.85 / 2.81 | 2.81 / 2.57 |
| cubic_large B / A | 2.57 / 2.65 | 2.65 / 3.37 |
| projected B / A | 3.37 / 3.26 | 3.26 / 3.01 |
| heat_s1 B / A | 3.01 / 10.61 | 10.61 / 30.68 |
| heat_s3 B / A | 30.68 / 28.79 | 28.79 / 73.88 |

External load was 2.5-3.4 throughout (three to five busy cores out of 192 from
other users: one `enzymexlamlir-opt`, three long-running `julia` processes, and
idle bazel servers). The rise from `heat_s1_B` onward is self-inflicted: the
heat cells are the only ones that hit threaded LAPACK dpotrf/dtrsm, so the
1-minute average tracks this process's own thread count. `pgrep -af taskset`
was checked before the first batch and after the last and found nothing, so no
sibling pinned single-core measurement on CPUs 60-63 overlapped this work. Load
before every batch start satisfied the load1 < 10 gate (4.21, 2.85, 4.71).

## Harness changes made for this measurement

- `cubic_kernel_dynamics.jl`: replaced the dead `optimize=:probprog_no_opt` /
  `:probprog` arm selector with heat's mechanism (`ARM` in A/B ->
  `DEBUG_PROBPROG_DISABLE_OPT`, `DEBUG_PROBPROG_DISABLE_SICM`,
  `CompileOptions(optimization_passes=:probprog,
  disable_licm_optimization_passes=..., sync=true)`); added a `small`/`large`
  size argument (D=60,T=50 / D=100,T=100) so both reported instances come from
  one file; added `Reactant.set_default_backend("cpu")`; added a fixed-seed
  verification call after the timed runs that dumps samples, accept decisions,
  divergence flags and log densities; added the IR report and HLO dump. Model
  body unchanged.
- `projected_dynamics.jl`: the same changes; its single size configuration
  (D=100, D_obs=30, T=100) is unchanged. Model body unchanged.
- `prelowering_common.jl` (new): shared arm knob table plus the IR counting and
  cell/HLO writing helpers. `count_in_out_loops` / `ir_report` are copied
  verbatim from `heat_inverse.jl` so every row is classified by identical code.
- `run_prelowering_batch.sh` (new): the batch driver; records the binary md5 and
  loadavg at batch start/end and per cell.
- `analyze_ir.py` (new): the depth-classified IR evidence tool described above.
- `aggregate.py` / `finalize.py` (new): build `cubic_projected_RAW_2026-08-24.json`
  from the batch log directories.
- `PRELOWER_NRUNS` env override on the two harnesses (default 5, i.e. the
  protocol) used only for the 25-run jitter diagnostics in `jitter_diag/`.
- `heat_inverse.jl`, `heat_inverse_data.jl`, `heat_inverse_v2.jl`: **not
  modified.** The heat cells were produced by the existing `bench` and `parity`
  modes.

## Files

- `cubic_projected_RAW_2026-08-24.json`: all 5 per-run timings for every cell
  of every batch, per-cell IR counts, verification samples/accept decisions,
  compile times, the batch metadata blocks, and the cross-batch summary.
- `logs_batch1/`, `logs_batch2/` (headline), `logs_batch3/`: per-cell run
  logs, `batch_meta.txt`, `cells/*.json`, `hlo/*.mlir`, `parity/*.json`.
- `jitter_diag/`: the 25-run warm-up-ramp diagnostics. Note that
  `benchmark/probprog/.gitignore` excludes `*.log`, so these three files and the
  per-cell run logs under `logs_batch*/` are on disk but will not be picked up
  by `git add`; the JSON, the MLIR dumps and `batch_meta.txt` will be.

## Repro

```
cd oopsla/Reactant.jl
bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch.sh
```

or per cell:

```
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/cubic_kernel_dynamics.jl <A|B> <small|large> bench
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/projected_dynamics.jl <A|B> bench
JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
  benchmark/probprog/ablation/pre_lowering/heat_inverse.jl <A|B> <1|3> <bench|parity>
```
