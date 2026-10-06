# AD-batching codegen-only ablation

Date: 2026-07-14

## Result

For the one-GPU Brusselator microbenchmark at `N=4096`, `K=8`, enabling only
AD differentiation batching makes the generated GPU program slower when Reactant's
generated post-Enzyme HLO optimization transforms are disabled:

| Configuration | Compile (s) | Steady median (ms) | Steady minimum (ms) | Throughput (Gcell-direction/s) | Per direction (ms) |
|---|---:|---:|---:|---:|---:|
| AD batching off | 20.370277 | 7.215861 | 7.011988 | 18.600 | 0.901983 |
| AD batching on | 20.448879 | 8.728857 | 8.544307 | 15.376 | 1.091107 |
| Batched / unbatched | 1.0039x | 1.2097x | 1.2185x | 0.8267x | 1.2097x |

The batched median is 1.512996 ms, or 20.97%, slower. Its minimum is 21.85% slower,
and throughput falls by 17.33%. The single compilation measurement differs by only 0.39%;
this run therefore shows no meaningful compilation-time improvement from batching, although
one compilation per variant is not enough for a compiler-latency conclusion.

This answers the narrow ablation question: without Reactant's post-Enzyme HLO cleanup,
AD batching alone is not a runtime optimization for this kernel. It does not establish
whether individual post-Enzyme transforms help or hurt; that requires the separate
`post_opt=true` comparison.

## What was disabled

Both variants use the default `optimization_passes=:all` pipeline and set:

```julia
Reactant.CompileOptions(;
    sync=true,
    ad_optimization_passes=(
        diff_batch ? Reactant.ADOptimizationOptions(; diff_batch=true) : false
    ),
    disable_post_enzyme_hlo_optimization_passes=true,
)
```

The only experimental difference is `ad_optimization_passes`:

- batching off runs the ordinary independent `enzyme.fwddiff` calls;
- batching on runs `enzyme-diff-batch` immediately followed by
  `enzyme-batch-to-stablehlo` before core Enzyme.

`disable_post_enzyme_hlo_optimization_passes=true` suppresses the generated
`enzyme-hlo-*` transform pipelines after core Enzyme. It deliberately retains the
semantics-only `enzyme-batch-to-stablehlo` legalizer, ordinary canonicalization/CSE,
lowering, and XLA's own compiler optimizations. Removing those would change legality or
the backend rather than isolate the AD optimization. All pre-Enzyme passes, core Enzyme,
XLA settings, inputs, synchronization, and device selection are otherwise identical.

## Microbenchmark configuration

- workload: compressed Brusselator Jacobian block;
- grid: `4096 x 4096` (`N=4096`);
- differentiation chunk: `K=8` one-hot tangent directions;
- logical work: 134,217,728 cell-directions per invocation;
- warmups: 5 synchronized invocations;
- samples: 30 synchronized invocations;
- device: GPU 0 only, NVIDIA GeForce RTX 5090, 32,607 MiB;
- driver/runtime/toolkit: NVIDIA 575.57.08, CUDA 12.9;
- Julia: 1.11.7;
- Reactant: 0.2.272 from repository commit
  `7203331aa7c4f20fc763837154646081425f0272`;
- Enzyme: 0.13.173.

The GPU was idle before the pair, and `CUDA_VISIBLE_DEVICES=0` exposed only one device to
the process. GPU clocks were not manually locked. The unbatched run was executed first;
the minimum-time result and the independent kernel profile below agree with the median,
so the conclusion is not dependent on a single timing statistic.

## Reproduction

Run from the Reactant repository root using the matching checkout library:

```sh
env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:$PWD:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/profile_k8.jl \
  --n=4096 --samples=30 --warmup=5 \
  --diff-batch=false --post-opt=false

env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:$PWD:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/profile_k8.jl \
  --n=4096 --samples=30 --warmup=5 \
  --diff-batch=true --post-opt=false
```

Fresh output from this run:

```text
K=8 N=4096 diff_batch=false post_opt=false compile=20.370277 s median=7.215861 ms minimum=7.011988 ms
K=8 N=4096 diff_batch=true  post_opt=false compile=20.448879 s median=8.728857 ms minimum=8.544307 ms
```

## Generated-code evidence

A separate 30-sample XProf capture with the same `N`, `K`, GPU, batching toggle, and
`post_opt=false` setting decomposes the device time:

| Configuration | Kernel | Average (ms) | Registers/thread | Occupancy |
|---|---|---:|---:|---:|
| AD batching off | `loop_concatenate_fusion` | 6.942093 | 38 | 100% |
| AD batching on | `loop_add_fusion` | 3.370381 | 40 | 100% |
| AD batching on | `loop_concatenate_fusion` | 5.077448 | 34 | 100% |

The batched kernel sum is 8.447829 ms, 21.69% above the unbatched 6.942093 ms, closely
matching the fresh 20.97% host-median regression. The thunk and memory plans show the
code-generation difference directly:

| Configuration | GPU kernels/invocation | Assigned memory | Extra derivative-sized temporary |
|---|---:|---:|---:|
| AD batching off | 1 | 4.25 GiB | none |
| AD batching on | 2 | 5.25 GiB | 1.00 GiB `f64[8,4096,4096]` |

The batched program computes the shared nonlinear tangent in `loop_add_fusion`, writes the
1 GiB value, and reads it in the terminal fusion. The unbatched program keeps each lane's
corresponding arithmetic inside one terminal fusion and materializes no derivative-sized
intermediate. The detailed HLO and profile analysis is in
[`compressed-jacobian-diff-batch-profile-report.md`](compressed-jacobian-diff-batch-profile-report.md).

The compact retained evidence is under `results/k8-n4096-regression/`:

- `profiles/{unbatched,batched}-kernel-stats.tsv`;
- `xla/{unbatched,batched}-after-optimizations.hlo`;
- `xla/{unbatched,batched}-memory-usage.txt`;
- `xla/{unbatched,batched}-thunk-sequence.txt`.

## Correctness check

The focused `N=16`, `K=8`, one-hot validation was run for both `diff_batch=false` and
`diff_batch=true`, with `post_opt=false` and the same GPU environment. Both passed the
primal comparison, ordinary and compiled Enzyme JVP comparisons, centered finite
difference comparison, compressed `(512, 8)` shape, finite-value, and non-aliasing checks.

## Conclusion

With post-Enzyme HLO transforms removed from both sides, AD batching changes the final
code from one fused GPU kernel to two kernels connected by a 1 GiB temporary. For this
microbenchmark it is a clear runtime loss, while compilation time is effectively neutral
in the observed pair. A useful batching cleanup must recover lane-local fusion and remove
the `f64[K,N,N]` temporary; batching by itself does not do so.
