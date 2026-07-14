# Brusselator batching without post-Enzyme HLO optimization

Date: 2026-07-14

## Outcome

This ablation disables only Reactant's generated post-Enzyme `enzyme-hlo-*` transforms.
The rest of the compiler pipeline, including inlining, canonicalization, CSE, lowering, and
XLA compilation, remains enabled. Both benchmark variants use the same option; their only
difference is whether `enzyme-diff-batch` is enabled.

All correctness checks pass. On the CPU backend, raw differentiation batching is slower for
every K where it fires: 2.57x at K=2, 2.58x at K=4, 2.63x at K=8, and 4.96x at K=12.
Therefore the earlier regressions cannot be attributed solely to the post-Enzyme HLO
optimizer. The CPU result does not establish how this ablation behaves on GPU.

## Exact pipeline ablation

The benchmark sets:

```julia
CompileOptions(;
    sync=true,
    ad_optimization_passes=diff_batch ?
        ADOptimizationOptions(; diff_batch=true) : false,
    disable_post_enzyme_hlo_optimization_passes=true,
)
```

The batching sequence remains:

```text
enzyme-diff-batch
enzyme-batch-to-stablehlo
enzyme
```

`enzyme-diff-batch` creates `enzyme.concat` and `enzyme.extract`, and the immediately
following legalizer converts those helpers before differentiation. On Brusselator, core
Enzyme creates another helper layer while differentiating batched slice/reshape operations.
With generated HLO transforms removed, that second layer must also be passed through the
semantics-only `enzyme-batch-to-stablehlo` legalizer; otherwise XLA rejects
`enzyme.extract` as an unsupported operation.

The option does not remove the whole `optimization_passes(...)` bundle. Post-Enzyme calls
retain inlining, SROA where applicable, canonicalization, and CSE, but omit
`enzyme-hlo-generate-td`, `transform-interpreter`, and `enzyme-hlo-remove-transform`.
Required dialect lowering and XLA's own compiler optimizations also remain enabled.

The focused pipeline tests confirm that the HLO-enabled and HLO-disabled configurations
contain the same numbers of `canonicalize` and `cse` passes. Enabling the ablation option
removes only the post-Enzyme half of the generated HLO transform segments.

## Environment and commands

- CPU: AMD Ryzen Threadripper PRO 7995WX
- Julia: 1.11.7
- Reactant: 0.2.272, working tree based on `b870933b49d3402a0edb786e29c779d09aac6ac9`
- Enzyme.jl: 0.13.173
- Backend: Reactant CPU, using the locally built `deps/ReactantExtra/bazel-bin/libReactantExtra.so`
- Performance problem: N=4096, K=1,2,4,8,12, one-hot seeds, 30 synchronous samples
- Validation problem: N=16, K=1,2,4,8,12, dense seeds

The local backend preference is inherited with:

```sh
export JULIA_LOAD_PATH="@:$PWD:@stdlib"
```

Validation and performance use the same commands for both configurations:

```sh
for diff_batch in false true; do
  julia --project=benchmark/brusselator --startup-file=no \
    benchmark/brusselator/runbenchmarks.jl \
    --backend=cpu --mode=validation --n=16 --ks=1,2,4,8,12 \
    --samples=1 --diff-batch=$diff_batch --post-opt=false

  julia --project=benchmark/brusselator --startup-file=no \
    benchmark/brusselator/runbenchmarks.jl \
    --backend=cpu --mode=performance --n=4096 --ks=1,2,4,8,12 \
    --samples=30 --diff-batch=$diff_batch --post-opt=false
done
```

Raw outputs are under `results/cpu-diff-batch-no-post-opt/`.

## Correctness

Both configurations pass every validation check with the same values:

- Reactant primal vs ordinary loop: max abs `5.684e-14`, max rel `2.109e-16`.
- Reactant JVP vs ordinary Enzyme: max abs `1.137e-13`, max rel `9.716e-15`.
- Every K chunk exactly matches its independently compiled JVP.
- Chunk vs finite-difference max absolute errors range from `4.105e-8` to `5.869e-8`.
- Every output is finite, has shape `(16, 16, 2)`, and uses an independent buffer.

Both N=4096 performance runs also pass their finite-output and independent-buffer sanity
checks.

## CPU performance

Steady time is the median of 30 synchronous samples. Slowdown is batched/unbatched; K=1 is
a control because `enzyme-diff-batch` has no repeated calls to combine there.

| K | Compile off (s) | Compile on (s) | Median off (ms) | Median on (ms) | Off per direction (ms) | On per direction (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.4495 | 2.3702 | 77.667 | 82.150 | 77.667 | 82.150 | 1.058x |
| 2 | 2.3423 | 2.3738 | 149.911 | 385.745 | 74.955 | 192.872 | 2.573x |
| 4 | 2.8978 | 2.6784 | 311.539 | 803.876 | 77.885 | 200.969 | 2.580x |
| 8 | 3.6627 | 3.3617 | 618.702 | 1624.090 | 77.338 | 203.011 | 2.625x |
| 12 | 4.2388 | 4.0139 | 908.876 | 4504.121 | 75.740 | 375.343 | 4.956x |

The unbatched per-direction time is flat at roughly 75-78 ms. Batched K=2/4/8 costs about
193-203 ms per direction, and K=12 rises to 375 ms per direction. The 5.8% K=1 difference
provides a run-to-run noise estimate, but it is much smaller than the K>=2 regressions.

Compile time changes range from 8.2% lower to 1.3% higher and are based on one observation
per kernel, so there is no strong compile-time conclusion.

## Relation to the prior GPU result

The earlier post-optimized GPU run improved K=12 by 3.35x while regressing K=2/4/8. This
CPU ablation instead regresses every batched width, including K=12. Since both backend and
post-Enzyme optimization differ, the two tables are not a direct optimizer A/B comparison.
The supported conclusion is narrower: without Reactant's post-Enzyme HLO transforms,
`enzyme-diff-batch` by itself does not improve this workload on the tested CPU.

The GPU ablation should be rerun with these same `--post-opt=false` commands once a working
GPU client is available.

## Focused verification

```sh
julia --project=. --startup-file=no test/core/compile_options.jl
```

The focused file passed 168 assertions. No repository-wide test suite was run.
