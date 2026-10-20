# Brusselator Newton step benchmark

This suite computes one Newton step for the two-dimensional Brusselator on a
periodic `N × N` grid. The state has two species and `2N²` `Float64` values.
The model uses the residual from the
[NonlinearSolve large systems tutorial](https://docs.sciml.ai/NonlinearSolve/v2.7/tutorials/large_systems/).

The application step computes the residual `r = F(x)`, forms the full dense
Jacobian `J`, solves `J * step = r` with LU factorization, and returns
`x - damping * step`. It uses `Enzyme.jacobian(Enzyme.Forward, ...; chunk=Val(1))`
to form the Jacobian. Thus each input direction starts as a separate AD request.
The compiler can combine compatible requests when derivative batching is enabled.

The suite measures three functions: the residual, the full Jacobian, and the
complete Newton step. Like the training step in `misc/bloch_rf_optimization.jl`,
the application benchmark measures one update at a fixed input. It does not run
a convergence loop. A step need not reduce the residual norm. The default damping
is one; the suite has no line search.

## Setup

Use Julia 1.11 or later. From the Reactant repository root:

```bash
export REACTANT_ROOT="$PWD"
BENCH="$REACTANT_ROOT/benchmark/brusselator"
export JULIA_LOAD_PATH="@:$REACTANT_ROOT:@stdlib"
julia --project="$BENCH" --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

The project uses this checkout through its `[sources]` entries. If an existing
Manifest refers to an older checkout, save a copy and update its dependencies:
`using Pkg; Pkg.Registry.update(); Pkg.update()`. Local package paths are retained.

For native compiler changes, first follow the
[local build guide](../../docs/src/tutorials/local-build.md). To use local core
Enzyme, pass `--extraopt=--override_repository=enzyme=/path/to/Enzyme/enzyme` to
`deps/build_local.jl`. The load path above makes the root native-library
preference visible. Start a new Julia process after each rebuild.

## Run

Start with a small grid and check the results before measuring performance.
To store logs and results outside the source tree:

```bash
set -euo pipefail
RUN="$REACTANT_ROOT/../llvmdev-results/brusselator/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
julia --project="$BENCH" --startup-file=no "$BENCH/validate.jl" \
  --backend=gpu --n=4 --ad=default 2>&1 | tee "$RUN/validation.log"
julia --project="$BENCH" --startup-file=no "$BENCH/runbenchmarks.jl" \
  --backend=gpu --n=8 --samples=30 --ad=default --output-dir="$RUN" \
  2>&1 | tee "$RUN/performance.log"
```

Use `--backend=cpu` for CPU execution. Set `CUDA_VISIBLE_DEVICES` before starting
Julia to select a GPU. The standard suite entry point also works:

```bash
julia --project="$BENCH" --startup-file=no benchmark/runbenchmarks.jl CUDA brusselator
```

Validation checks the residual against the scalar stencil and the Jacobian
against its analytic derivative. It also checks finite differences, the Newton
update, the linear solve residual, and input preservation.

`N=4` gives a `32 × 32` Jacobian. `N=8` gives a `128 × 128` Jacobian.
`N=16` gives a `512 × 512` Jacobian. Use `N ≥ 3`; the initial Jacobian for
`N=2` is singular.
Dense Jacobian storage grows as `N⁴`; increase the grid size with care.
The old large grids for a few derivative columns do not fit this workload.

## AD settings and measurements

All configurations use the normal `:all` optimization pipeline. `--ad` selects
which additional AD passes run:

| Value | Configuration | AD passes |
| --- | --- | --- |
| `default` | Default | None |
| `activity` | Activity | Activity refinement |
| `batch` | DiffBatch | Derivative batching |
| `hoist` | RegionHoist | Region hoisting |
| `all` | AllAD | All three |
| `compare` | All five configurations | As listed above |

Use `--ad=compare` with both validation and performance commands to compare these
configurations. A pass being enabled does not show that it changed the generated
code or improved performance. Inspect the IR and compare measured times.

The suite reuses `BenchmarkConfiguration` and `run_benchmark!` from
`benchmark/misc/common.jl`, as the Bloch benchmark does. Each Reactant measurement
compiles the function, executes one warmup, and profiles the requested repetitions
with `Reactant.Profiler.profile_with_xprof`. Inputs are prepared and transferred
before profiling. Each Newton measurement starts from the same state. Jacobian
creation, LU factorization, and the state update are part of the Newton step
measurement. Runtime is XProf's mean step time. On the CPU backend, the shared
runner also measures ordinary Julia execution with Chairmarks.

The shared `benchmark/utils.jl` utilities print a table and write
`brusselator_CUDAbenchmarks.json` and `brusselator_CUDAbenchmarks_tflops.json`
(or the equivalent backend name). Missing FLOP metrics use `-1`. Profiler traces
use Reactant's default scratch directory. Use a new output directory for each
run; the JSON filenames are reused. Validation has no performance timing.

Arguments use `--name=value`. Defaults are `--n=8` (`4` for validation), `--damping=1.0`,
`--samples=30`, `--ad=default`, `--backend=auto`, and the suite's `results/`
directory for `--output-dir`. Damping must be in `(0, 1]`. Explicit `--backend`
takes priority over `BENCHMARK_GROUP`; `auto` uses the shared backend selection.
`--mode=validation` also runs validation through `runbenchmarks.jl`.
