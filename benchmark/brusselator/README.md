# Brusselator benchmark

This suite measures a two-dimensional Brusselator residual and Enzyme forward-mode
Jacobian-vector products compiled by Reactant. It contains two complementary modes:

- `validation` uses a small grid and checks the Julia and Reactant primal residuals,
  ordinary and compiled Enzyme JVPs, centered finite differences, independent chunked
  JVP calls, finite values, output shapes, and buffer aliasing.
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
  --mode=validation --n=32 --ks=1,2,4,8,12 --samples=10 --diff-batch=true
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
  --mode=performance --n=2048 --ks=1,2,4 --samples=30 --diff-batch=true
```

The output separates compilation, first execution, steady median, and steady minimum
times. It also reports throughput, per-direction time, logical-state size, and the
argument-buffer footprint for each chunk.

## Command-line options

All options use `--name=value` syntax:

| Option | Values | Validation default | Performance default |
|---|---|---:|---:|
| `--mode` | `validation`, `performance` | `validation` | `performance` |
| `--n` | grid width greater than 1 | `16` | `4096` |
| `--ks` | comma-separated subset of `1,2,4,8,12` | `1,2,4` | `1,2,4,8,12` |
| `--seed` | `dense`, `onehot` | `dense` | `onehot` |
| `--samples` | positive integer | `5` | `30` |
| `--epsilon` | positive floating-point value | `1e-4` | unused |
| `--diff-batch` | `true`, `false` | `false` | `false` |

## Layout and compiler experiments

- `workload.jl` contains the initialization, scalar Julia reference, Reactant-compatible
  residual, Enzyme forward JVP, and statically independent chunk wrappers. The AD-facing
  residual is a pure function: every JVP in a chunk receives the same primal state,
  coordinates, and parameters, and only its tangent seed and destination differ.
- `runbenchmarks.jl` contains correctness checks, compilation, synchronized timing, and
  command-line handling.
- `inspect_mlir.jl` saves the initial, pre-batching, post-batching, and legalized MLIR
  for each selected chunk size under the ignored `results/` directory. It also applies
  a diagnostic post-core-Enzyme helper legalization explicitly and saves both sides of
  it; that diagnostic stage is not part of the production pass ordering.
- `profile_k8.jl` compiles only the K=8 chunk and can save XProf kernel and framework-op
  reports without profiling the primal and single-JVP setup cases.
- `Project.toml` is the isolated benchmark environment.

`compile_timed` in `runbenchmarks.jl` is the central call to `Reactant.compile`. Set
`--diff-batch=true` to compile with
`ADOptimizationOptions(; diff_batch=true)`, or leave the default `false` to disable all
optional AD-aware optimizations. All other compile and benchmark settings are identical.
Run validation for both configurations before comparing performance results.

To inspect differentiation batching at each relevant pipeline stage, run:

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/inspect_mlir.jl \
  --n=3 --ks=1,2,4,8,12 --output-dir=benchmark/brusselator/results/mlir
```

The inspection grid is intentionally small because the operation structure and batching
decision do not depend on the tensor dimensions.
