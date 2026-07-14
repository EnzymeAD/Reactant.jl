# Brusselator `enzyme-diff-batch` evaluation

Date: 2026-07-14

## Outcome

After the minimal compiler fixes described below, differentiation batching fires for every
repeated-call case. K=2,4,8,12 each lower from K width-1 `enzyme.fwddiff` operations to one
operation of width K. K=1 correctly remains one width-1 operation.

All off/on correctness checks pass with identical numerical results. At N=4096, the pass is
neutral at K=1, regresses K=2/4/8, and improves K=12 from 84.927 ms to 25.356 ms (3.349x).

## Environment

- GPU: NVIDIA GeForce RTX 5090, CUDA compute capability 12.0a
- Driver/runtime/toolkit: CUDA 12.9 / 12.9 / 12.9
- Julia: 1.11.7
- Reactant: 0.2.272, repository SHA `bafdcd75390da5dc415d870f17b17d8445424bf2`
- Enzyme.jl: 0.13.173
- Enzyme source: `b4b8c3984ce99c659481d9babfca6f0ced5f917b`
- Backend: locally built `deps/ReactantExtra/bazel-bin/libReactantExtra.so`

Pass off uses `ad_optimization_passes=false`. Pass on uses
`ADOptimizationOptions(; diff_batch=true)`. Both use
`CompileOptions(; sync=true, ...)`; the workload, inputs, synchronization, sizes, and timing
method are otherwise identical.

The isolated benchmark environment inherited the root environment's local Reactant_jll
preference with:

```sh
export JULIA_LOAD_PATH="@:$PWD:@stdlib"
export CUDA_VISIBLE_DEVICES=0
```

## Commands

```sh
julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/inspect_mlir.jl \
  --n=3 --ks=1,2,4,8,12 \
  --output-dir=benchmark/brusselator/results/mlir-final

julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=validation --n=16 --ks=1,2,4,8,12 --samples=5 --diff-batch=false

julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=validation --n=16 --ks=1,2,4,8,12 --samples=5 --diff-batch=true

julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=performance --n=4096 --ks=1,2,4,8,12 --samples=30 --diff-batch=false

julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=performance --n=4096 --ks=1,2,4,8,12 --samples=30 --diff-batch=true
```

Focused regression commands:

```sh
julia --project=. --startup-file=no test/core/compile_options.jl

cd deps/ReactantExtra
bazel test [the local CUDA/Clang/Enzyme override build flags] \
  @enzyme//test/MLIR:OptimizeAD_fwd_batch.mlir.test
```

No repository-wide test suite was run.

## Initial structural blocker and minimal fixes

`BatchDiffCacheKey` groups operations by callee symbol, primal SSA inputs, activities, and
block. Initially every repeated Brusselator JVP had a unique callee symbol and threaded
unchanged primal results into the next call, so every compatibility group had size one.

The minimal Reactant changes are:

1. Cache a traced Enzyme primal callee for the duration of one compilation, allowing
   repeated calls with the same structural signature to reference one symbol.
2. Ask the nested primal trace to return only actually mutated arguments instead of every
   unchanged primal argument.
3. Detect mutation relative to the arguments after Reactant's input transpose/padding
   preparation. The old comparison against raw block arguments falsely classified those
   preparation operations as user mutation.
4. Legalize Enzyme concat/extract helpers both before and after core Enzyme. The post step
   is necessary because differentiating the newly batched call creates another helper layer.

After those changes, K=4 contains one primal symbol and four compatible calls with the same
primal operands `%arg8`, `%arg9`, and `%arg18`; only the two tangent operands differ.

At that point the pass itself crashed. Its merge loop indexed the three-element primal input
array with the five-element interleaved primal/tangent operand index. Brusselator's activity
pattern is exactly `dup, dup, const`, so the old code selected the wrong second primal and
then read out of bounds. The Enzyme fix uses the primal-activity index for `key.inputs` in
both the forward and reverse merge loops. A focused forward test now covers two duplicated
inputs followed by a constant.

No Brusselator workload code or solver method was changed. No canonicalization or region
hoisting experiment was enabled.

## MLIR result

The saved MLIR is under `results/mlir-final/k{K}/`. "Helpers" are
`enzyme.concat / enzyme.extract`; "legal additions" are the StableHLO concatenate/slice
increase caused immediately by helper legalization.

| K | Before batching | After batching | Fired? | Helpers after pass | Legal additions | Helpers after full lowering |
|---:|---|---|---|---:|---:|---:|
| 1 | 1 x width 1 | 1 x width 1 | no repeated group | 0 / 0 | +0 / +0 | 0 / 0 |
| 2 | 2 x width 1 | 1 x width 2 | yes | 2 / 4 | +2 / +4 | 0 / 0 |
| 4 | 4 x width 1 | 1 x width 4 | yes | 2 / 8 | +2 / +8 | 0 / 0 |
| 8 | 8 x width 1 | 1 x width 8 | yes | 2 / 16 | +2 / +16 | 0 / 0 |
| 12 | 12 x width 1 | 1 x width 12 | yes | 2 / 24 | +2 / +24 | 0 / 0 |

Immediately after helper legalization, the module has one `fwddiff` with the expected width
and no Enzyme helpers. Full off/on StableHLO concatenate/slice counts are 8/16 vs 8/16,
16/32 vs 26/40, 32/64 vs 42/80, 64/128 vs 74/160, and 96/192 vs 106/240 for
K=1,2,4,8,12 respectively. Fully lowered off/on modules differ for K>1, confirming the
optimization survives the pipeline.

## Correctness

Both configurations pass every N=16 validation check with the same values:

- Julia and Reactant primal residual maximum absolute error: `0.0`.
- Ordinary Enzyme JVP vs centered finite difference: max abs `4.105e-8`, max rel
  `6.538e-8`.
- Reactant JVP vs ordinary Enzyme: max abs `1.137e-13`, max rel `4.858e-15`.
- Each chunk output vs its independently compiled JVP: max absolute and relative error
  `0.0` for K=1,2,4,8,12.
- Chunk vs finite-difference max absolute errors: `4.105e-8`, `4.720e-8`, `5.339e-8`,
  `5.339e-8`, and `5.869e-8` for K=1,2,4,8,12.
- Every output is finite, has shape `(16, 16, 2)`, and uses an independent buffer.

Both N=4096 performance runs also pass finite-output and independent-buffer sanity checks.

## Performance

Steady time is the median of 30 synchronous samples at N=4096. Speedup is off/on; values
below 1 are regressions. Compile time is one observed compilation per configuration.

| K | Compile off (s) | Compile on (s) | Steady off (ms) | Steady on (ms) | Speedup | Runtime effect |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.2207 | 2.0610 | 0.571 | 0.567 | 1.007x | 0.7% faster |
| 2 | 2.2633 | 2.1646 | 1.137 | 2.088 | 0.545x | 83.6% slower |
| 4 | 2.5808 | 2.4847 | 2.243 | 7.141 | 0.314x | 218.4% slower |
| 8 | 3.4223 | 3.2339 | 6.255 | 16.198 | 0.386x | 159.0% slower |
| 12 | 4.3168 | 4.2733 | 84.927 | 25.356 | 3.349x | 70.1% faster |

The compile-time observations are 1.0-7.2% lower with batching, but a single compile per
case is not enough to separate that from process variation. The steady-state conclusion is
unambiguous: wider differentiation hurts the smaller repeated-call cases on this GPU, but
avoids the severe K=12 degradation and yields a 3.35x speedup there.
