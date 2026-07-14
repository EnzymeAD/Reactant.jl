# Brusselator benchmark

This suite adapts NonlinearSolve.jl's sparse Brusselator Jacobian construction. It measures
the two-dimensional residual and Enzyme forward-mode Jacobian-vector products compiled by
Reactant. A chunk of `K` colored directions produces one compressed Jacobian block with
shape `(2N^2, K)`, matching the data consumed by sparse Jacobian decompression instead of
materializing `K` separate output states. It contains two complementary modes:

- `validation` uses a small grid and checks the Julia and Reactant primal residuals,
  ordinary and compiled Enzyme JVPs, centered finite differences, the compressed block
  against independently computed JVP columns, finite values, output shapes, and buffer
  aliasing.
- `performance` uses a large grid and synchronized device execution. It omits finite
  differences because their cancellation error grows with the `(N - 1)^2` diffusion
  scaling, but retains finite-output and non-aliasing sanity checks.

The benchmark uses one unsharded Reactant device. It does not request distributed
execution, even when multiple GPUs are visible.

## Setup

Run the following commands from the Reactant repository root with Julia 1.11 or later:

```sh
julia --project=benchmark/brusselator --startup-file=no -e \
  'using Pkg; Pkg.instantiate()'
```

The suite's `Project.toml` resolves Reactant from this repository checkout and installs
the compatible Enzyme version. No package installation is needed on subsequent runs
unless the environment changes.

When the repository's `LocalPreferences.toml` selects a locally built `Reactant_jll`,
also put the repository project on Julia's load path before running the benchmark from
the repository root:

```sh
export JULIA_LOAD_PATH="@:$PWD:@stdlib"
```

This makes the benchmark project inherit the matching local `libReactantExtra`. Without
it, Julia can load the released artifact library alongside this checkout's compiler code,
which is an unsupported ABI mismatch.

To select one GPU on a multi-GPU machine, set `CUDA_VISIBLE_DEVICES` when launching Julia.
For example, all commands below select device 0.

## Correctness validation

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl --mode=validation
```

Validation defaults to `N=16`, dense tangent seeds, chunk sizes `K=1,2,4`, and five
synchronous timing samples. A larger correctness case or the optional chunk sizes can be
selected explicitly:

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=validation --n=32 --ks=1,2,4,8,12 --samples=10 \
  --diff-batch=true --post-opt=false
```

## Performance benchmark

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl --mode=performance
```

Performance mode defaults to a `4096 x 4096` grid, one-hot tangent seeds, chunk sizes
`K=1,2,4,8,12`, and 30 synchronized timing samples. On a device with less memory, start
with a smaller grid or fewer simultaneous JVPs:

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=performance --n=2048 --ks=1,2,4 --samples=30 \
  --diff-batch=true --post-opt=false
```

The output separates compilation, first execution, steady median, and steady minimum
times. It also reports throughput, per-direction time, logical-state size, and the
argument-buffer footprint for each chunk. The timed chunk ends after filling the compressed
Jacobian block; sparse decompression and the Newton linear solve are deliberately outside
the measurement.

## Command-line options

All options use `--name=value` syntax:

| Option | Values | Validation default | Performance default |
|---|---|---:|---:|
| `--mode` | `validation`, `performance` | `validation` | `performance` |
| `--backend` | `auto`, `cpu`, `gpu` | `auto` | `auto` |
| `--n` | grid width greater than 1 | `16` | `4096` |
| `--ks` | comma-separated subset of `1,2,4,8,12` | `1,2,4` | `1,2,4,8,12` |
| `--seed` | `dense`, `onehot` | `dense` | `onehot` |
| `--samples` | positive integer | `5` | `30` |
| `--epsilon` | positive floating-point value | `1e-4` | unused |
| `--diff-batch` | `true`, `false` | `false` | `false` |
| `--post-opt` | `true`, `false` | `true` | `true` |

## Layout and compiler experiments

- `workload.jl` contains the initialization, scalar Julia reference, Reactant-compatible
  residual, Enzyme forward JVP, and statically independent chunk wrappers. The AD-facing
  residual is a pure function: every JVP in a chunk receives the same primal state,
  coordinates, and parameters, and only its tangent seed differs. The results are flattened
  into adjacent columns of one `(2N^2, K)` compressed Jacobian block, following the
  DifferentiationInterface sparse-Jacobian staging used by NonlinearSolve.jl.
- `runbenchmarks.jl` contains correctness checks, compilation, synchronized timing, and
  command-line handling.
- `inspect_mlir.jl` saves the initial, pre-batching, post-batching, and legalized MLIR
  for each selected chunk size under the ignored `results/` directory. It also applies
  post-core-Enzyme helper legalization explicitly and saves both sides of it. The normal
  optimized pipeline does not add that pass explicitly; the `--post-opt=false` ablation
  does because the omitted HLO transforms otherwise leave helper ops illegal for XLA.
- `profile_k8.jl` compiles only the K=8 chunk and can save XProf kernel and framework-op
  reports without profiling the primal and single-JVP setup cases. Its
  `--post-opt=false` ablation preserves the normal pre-Enzyme passes and required
  legalization but skips the post-Enzyme StableHLO optimization pipeline.
- `regressions/brusselator-k8-n4096-batched.mlir` is the exact StableHLO module captured
  by XLA from the profiled `N=4096`, `K=8`, `diff_batch=true`, `post_opt=false` run. It is
  a full production-size dump, not a hand-written or minimized example.
- `regressions/brusselator-slice-elementwise-concat-min.mlir` keeps the production
  `8x4096x4096` types and all eight partitions, but reduces the graph to the two tangent
  concatenations, one shared elementwise value, its two consumers, and their lane slices.
- `Project.toml` is the isolated benchmark environment.

`compile_timed` in `runbenchmarks.jl` is the central call to `Reactant.compile`. Set
`--diff-batch=true` to compile with
`ADOptimizationOptions(; diff_batch=true)`, or leave the default `false` to disable all
optional AD-aware optimizations. All other compile and benchmark settings are identical.
Run validation for both configurations before comparing performance results.

To measure batching without the post-Enzyme HLO optimizer, run the K=8 profiler for
both batching configurations:

```sh
for diff_batch in false true; do
  julia --project=benchmark/brusselator --startup-file=no \
    benchmark/brusselator/profile_k8.jl \
    --diff-batch=$diff_batch --post-opt=false
done
```

The `--post-opt=false` path sets
`CompileOptions(disable_post_enzyme_hlo_optimization_passes=true)`. It leaves the normal
`:all` pipeline and the `enzyme-diff-batch` -> `enzyme-batch-to-stablehlo` ordering intact.
Only the generated post-Enzyme `enzyme-hlo-*` transforms are omitted; inlining, SROA,
canonicalization, CSE, lowering, and XLA's own compiler optimizations remain enabled and
identical in both configurations. A semantics-only `enzyme-batch-to-stablehlo` legalization
also runs after core Enzyme because core differentiation can create a second generation of
`enzyme.concat` and `enzyme.extract` helpers; without it, the no-HLO-opt IR is not legal for
XLA.

The compressed-Jacobian result on one RTX 5090 is recorded in
[`compressed-jacobian-diff-batch-gpu-report.md`](compressed-jacobian-diff-batch-gpu-report.md),
the focused no-post-Enzyme-optimization ablation is in
[`ad-batching-codegen-ablation-report.md`](ad-batching-codegen-ablation-report.md),
and its kernel/IR root-cause analysis is in
[`compressed-jacobian-diff-batch-profile-report.md`](compressed-jacobian-diff-batch-profile-report.md).
The exact StableHLO input for an Enzyme-JAX regression test is
[`regressions/brusselator-k8-n4096-batched.mlir`](regressions/brusselator-k8-n4096-batched.mlir).
The structurally minimized reproducer is
[`regressions/brusselator-slice-elementwise-concat-min.mlir`](regressions/brusselator-slice-elementwise-concat-min.mlir).

Run the current Enzyme-JAX StableHLO optimizer on the minimized input with:

```sh
enzymexlamlir-opt \
  benchmark/brusselator/regressions/brusselator-slice-elementwise-concat-min.mlir \
  --enzyme-hlo-opt
```

The output still contains two `stablehlo.concatenate` operations, the three elementwise
operations on `tensor<8x4096x4096xf64>`, and all sixteen terminal slices. Run it on the
unmodified captured input with:

```sh
enzymexlamlir-opt \
  benchmark/brusselator/regressions/brusselator-k8-n4096-batched.mlir \
  --enzyme-hlo-opt
```

To inspect differentiation batching at each relevant pipeline stage, run:

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/inspect_mlir.jl \
  --n=3 --ks=1,2,4,8,12 --output-dir=benchmark/brusselator/results/mlir
```

The inspection grid is intentionally small because the operation structure and batching
decision do not depend on the tensor dimensions.
