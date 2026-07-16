# Compressed Brusselator Jacobian batching on one GPU

Date: 2026-07-14

## Question

This experiment measures only the effect of `enzyme-diff-batch` on the Brusselator
forward-mode Jacobian workload. The benchmark now follows the sparse-Jacobian staging used
by NonlinearSolve more closely: `K` colored JVPs terminate in one logical compressed
Jacobian block with shape `(2N^2, K)`. It does not expose `K` separate JVP output buffers,
and the timed region stops before sparse decompression and the Newton linear solve.

Post-Enzyme HLO optimization is disabled in both configurations. The normal pre-Enzyme
pipeline, `enzyme-diff-batch` -> `enzyme-batch-to-stablehlo` legalization, core Enzyme
lowering, required helper legalization, XLA compilation, input layout, and synchronization
are otherwise identical.

## Environment and commands

- GPU: one NVIDIA GeForce RTX 5090, selected with `CUDA_VISIBLE_DEVICES=0`
- XLA CUDA backend: compute capability 12.0a, CUDA driver/runtime/toolkit 12.9
- Julia 1.11.7
- Reactant 0.2.272 at repository commit `c7468c2a88d01dd8b0b41737658772385d1085ea`
- Enzyme 0.13.173
- Grid: `4096 x 4096` (`16,777,216` cells)
- Seeds: one-hot
- Chunk sizes: `K = 1, 2, 4, 8, 12`
- Samples: 30 synchronous steady-state executions per chunk
- One logical state: 0.250 GiB

The local ReactantExtra build was selected through the repository environment so the
BatchDiff mixed-activity input-indexing fix was present:

```sh
env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:/mnt/vimarsh6739/hiord/Reactant.jl:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=performance --backend=gpu --n=4096 --ks=1,2,4,8,12 \
  --samples=30 --seed=onehot --diff-batch=false --post-opt=false

env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:/mnt/vimarsh6739/hiord/Reactant.jl:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/runbenchmarks.jl \
  --mode=performance --backend=gpu --n=4096 --ks=1,2,4,8,12 \
  --samples=30 --seed=onehot --diff-batch=true --post-opt=false
```

## Timing results

The ratio is `batched / unbatched`; values above 1 are regressions. Runtime is the median
of 30 synchronized executions.

| K | Argument buffers (GiB) | Unbatched compile (s) | Batched compile (s) | Compile ratio | Unbatched runtime (ms) | Batched runtime (ms) | Runtime ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.750 | 3.3192 | 3.3863 | 1.020x | 0.619 | 0.613 | 0.990x |
| 2 | 1.250 | 2.4285 | 2.4131 | 0.994x | 1.501 | 2.258 | 1.504x |
| 4 | 2.250 | 3.1701 | 3.0995 | 0.978x | 3.048 | 4.390 | 1.440x |
| 8 | 4.250 | 4.5749 | 4.7711 | 1.043x | 7.269 | 8.782 | 1.208x |
| 12 | 6.250 | 5.1740 | 6.5974 | 1.275x | 10.316 | 12.395 | 1.202x |

Steady-state minimum times show the same result rather than a median-only outlier:

| K | Unbatched minimum (ms) | Batched minimum (ms) | Ratio |
|---:|---:|---:|---:|
| 1 | 0.608 | 0.606 | 0.997x |
| 2 | 1.487 | 2.144 | 1.442x |
| 4 | 2.981 | 4.317 | 1.448x |
| 8 | 7.228 | 8.737 | 1.209x |
| 12 | 10.109 | 12.276 | 1.214x |

The K=1 controls agree within about 1%, which supports treating the larger differences as
effects of the batching transformation rather than a mismatch between the two runs.

## Correctness

Validation ran on the same single GPU at `N=16` for every supported K with both
`--diff-batch=false` and `--diff-batch=true`, again with `--post-opt=false`.

- Every terminal block had shape `(512, K)`, i.e. `(2N^2, K)`.
- Every block matched the corresponding independently compiled JVP columns exactly:
  maximum absolute and relative error were both zero.
- Centered finite-difference maximum absolute errors ranged from `4.105e-8` to
  `5.869e-8`, within the benchmark tolerance.
- Finite-value and non-aliasing checks passed for every K.
- A K=12 MLIR audit at N=3 found one aliased terminal argument/result,
  `tensor<12x18xf64>`. Reactant's internal layout reverses the Julia dimensions, so this is
  the logical `(18, 12)` block; there are no 12 separate JVP output results.

## Interpretation

The current batching transformation is a runtime regression for this compressed Jacobian
workload on the RTX 5090. The penalty is largest at K=2 (50.4%) and K=4 (44.0%), then
settles near 20% at K=8 and K=12. It is not offset by a general compilation-time win:
compile time is approximately neutral through K=8, while K=12 is 27.5% slower to compile.

Consequently, these measurements do not justify enabling batching for this workload on the
basis of either runtime or compilation time. They instead provide a clean target for a
post-batching cleanup: eliminate or fuse the slice/elementwise/concat assembly introduced
when the wide JVP is partitioned into the terminal compressed block. The follow-up
[`compressed-jacobian-diff-batch-profile-report.md`](compressed-jacobian-diff-batch-profile-report.md)
confirms the mechanism with GPU profiles, final XLA HLO, buffer assignment, and a causal
ablation.

One current pipeline limitation is intentionally outside this ablation: with post-Enzyme
HLO optimization enabled, core Enzyme can produce a second generation of `enzyme.concat`
and `enzyme.extract` helpers in this compressed-output graph, and the current optimized
pipeline can leave them illegal for XLA. The batching-only comparison therefore uses the
working, matched `--post-opt=false` CompileOptions path and should not be read as the final
performance of a future cleanup-enabled pipeline.
