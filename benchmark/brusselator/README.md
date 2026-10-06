# Brusselator derivative baseline

This application-derived kernel evaluates a two-dimensional Brusselator residual and
independent Enzyme forward JVPs at one shared primal state. A chunk of `K` directions
produces a dense output block of shape `(2N², K)`. It does not perform coloring, sparse
Jacobian decompression, or a nonlinear solve. The scalar Julia residual, ordinary Enzyme
JVPs, finite differences, and individual compiled JVPs are independent references.

## Build and setup

From this Reactant checkout, with Julia 1.11 or later:

```sh
julia --project=deps --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=deps --startup-file=no deps/build_local.jl \
  --backend=cuda12 --jobs=48 --cache --cc=/usr/bin/clang-18
julia --project=benchmark/brusselator --startup-file=no -e \
  'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(path="lib/ReactantCore")]); Pkg.instantiate()'
export JULIA_LOAD_PATH="@:$PWD:@stdlib"
```

Use the Enzyme-JAX revision pinned by `deps/ReactantExtra/WORKSPACE` and its pinned core
Enzyme, without local repository overrides. The old July measurements used a local core
Enzyme override; they are historical results, not evidence for this baseline.
`build_local.jl` writes the root `LocalPreferences.toml` selecting the built library.
The load path above makes that preference visible to the benchmark environment. Start
fresh Julia processes after rebuilding. For other compilers and devices, see the
[local build guide](../../docs/src/tutorials/local-build.md).

## L1 validation

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/validate_l1.jl --backend=gpu --output-dir=/tmp/brusselator-l1
```

This checks all 20 combinations of batching disabled/enabled, dense/one-hot seeds, and
`K=1,2,4,8,12` at `N=16`. Each block must have shape `(512, K)`, match ordinary Enzyme,
independently compiled JVP columns, and finite differences, and preserve its input state
and seeds. Outputs must be finite, host buffers must not alias, and batching on/off must
agree. It writes a numerical TSV and runtime metadata, including the loaded native
library's real path and SHA256. `--backend=cpu` provides partial validation only.

To run one setting:

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl --mode=validation --backend=gpu \
  --n=16 --ks=1,2,4,8,12 --seed=dense --diff-batch=true --samples=1
```

Normal post-Enzyme HLO optimization remains enabled. `--post-opt=false` is rejected
explicitly: restoring that ablation while retaining required lowering belongs to L4.
Batch helpers are lowered both after request batching and inside the core Enzyme
postpasses, as required by current main.

## Production IR

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/inspect_mlir.jl --backend=gpu --n=16 --ks=1,2,4,8,12 \
  --output-dir=/tmp/brusselator-l1-ir
```

Use an empty output directory. The script compiles and executes each chunk with the
normal `:all` production pipeline and both batching settings, captures its input IR,
pass pipeline, and generated IR, then replays the exact captured prefix to expose the
AD boundary. No obsolete hand-assembled optimization pipeline is used. Assertions
require `K` width-one calls to the same callee before batching and one width-`K` call
after batching, followed by required helper lowering. The complete off/on programs
must execute and agree. Stage IR and a request-count TSV are saved beside the original
compiler dumps. Intermediate stage files are prefix replays of the captured production
pipeline; the original `*_pre_all_pm.mlir` and `*_post_all_pm.mlir` are direct dumps.

## Existing performance tools

`runbenchmarks.jl --mode=performance` and `profile_k8.jl` remain available for L4.
Performance mode defaults to `N=4096`, one-hot seeds, all five chunk widths, and 30
synchronized samples. It separates compilation, first execution, and steady execution.
Large-grid runs are not part of L1; start with `K=8` and account for device memory in L4.
The output block includes storage of all JVP columns. A smaller kernel count alone is
not a speedup claim.

```sh
CUDA_VISIBLE_DEVICES=0 julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/profile_k8.jl --n=4096 --diff-batch=true --post-opt=true \
  --profile-dir=/tmp/brusselator-profile
```

All arguments use `--name=value` syntax. Validation defaults to `N=16`, dense seeds,
`K=1,2,4,8,12`, five samples, finite-difference epsilon `1e-4`, and batching disabled.
`--backend` selects `auto`, `cpu`, or `gpu` in `runbenchmarks.jl`. The full L1 runner
and IR inspection default to `gpu` so CPU fallback cannot count as GPU completion.

## Files and historical evidence

- `workload.jl`: unchanged numerical workload from benchmark snapshot `c084f24f4`.
- `runbenchmarks.jl`: individual validation/performance configurations.
- `validate_l1.jl`: complete small-grid validation matrix and runtime provenance.
- `inspect_mlir.jl`: production compilation and exact-prefix IR inspection.
- `profile_k8.jl`: K=8 profiler entry point.
- `regressions/`: retained July minimal and full StableHLO examples for L5.
- `compressed-jacobian-diff-batch-gpu-report.md`,
  `compressed-jacobian-diff-batch-profile-report.md`, and
  `ad-batching-codegen-ablation-report.md`: historical July measurements and pipeline
  descriptions. Their old options and dependency overrides do not describe this port.

Current L1 source/build identities, commands, test outcomes, and artifact paths are
recorded separately in the workspace's `llvmdev-results/L1.md`.
