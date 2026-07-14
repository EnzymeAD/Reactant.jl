# Why differentiation batching slows the compressed Brusselator Jacobian

Date: 2026-07-14

## Finding

At K=8, differentiation batching turns one fused GPU kernel into two kernels and
materializes a 1.00 GiB `f64[8,4096,4096]` temporary. The temporary is the tangent of the
shared nonlinear term `u^2*v`; it is consumed once with a plus sign by the first species
and once with a minus sign by the second species. XLA chooses to compute that multi-use
wide value once and store it rather than clone it into the terminal output fusion.

This is the reason for the measured slowdown. Precomputing the shared term makes the final
kernel 1.865 ms faster, but the producer kernel costs 3.370 ms, leaving a measured 1.506 ms
net regression. The result is not a kernel-launch-overhead problem: both kernels process
hundreds of millions of values, and the extra kernel accounts for 39.9% of batched device
time.

## Matched profile

Both variants used one NVIDIA GeForce RTX 5090, `N=4096`, `K=8`, one-hot seeds, five
warmups, 30 synchronous samples, and `--post-opt=false`. Only `diff_batch` changed.

```sh
env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:/mnt/vimarsh6739/hiord/Reactant.jl:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/profile_k8.jl \
  --n=4096 --samples=30 --warmup=5 --diff-batch=false --post-opt=false \
  --profile-dir=benchmark/brusselator/results/compressed-gpu-profile/k8-off

env CUDA_VISIBLE_DEVICES=0 \
  JULIA_LOAD_PATH=@:/mnt/vimarsh6739/hiord/Reactant.jl:@stdlib \
  julia --project=benchmark/brusselator --startup-file=no \
  benchmark/brusselator/profile_k8.jl \
  --n=4096 --samples=30 --warmup=5 --diff-batch=true --post-opt=false \
  --profile-dir=benchmark/brusselator/results/compressed-gpu-profile/k8-on
```

| Variant | Compile (s) | Host median (ms) | Host minimum (ms) | Kernel count/execution |
|---|---:|---:|---:|---:|
| Unbatched | 19.8895 | 7.8989 | 7.3636 | 1 |
| Batched | 20.2277 | 9.3572 | 8.8252 | 2 |
| Batched / unbatched | 1.017x | 1.185x | 1.198x | 2x |

The XProf/CUPTI kernel aggregation gives the more useful decomposition:

| Variant | Kernel | Average (ms) | Registers/thread | Grid | Block |
|---|---|---:|---:|---:|---:|
| Unbatched | `loop_concatenate_fusion` | 6.942 | 38 | 524288 | 128 |
| Batched | `loop_add_fusion` | 3.370 | 40 | 262144 | 128 |
| Batched | `loop_concatenate_fusion` | 5.077 | 34 | 524288 | 128 |

The summed batched device time is 8.448 ms versus 6.942 ms unbatched, a 21.7% regression.
This agrees with both the profile's synchronized host timing and the earlier full benchmark.
Occupancy is reported as 100% for all three kernels, so reduced theoretical occupancy or
register pressure is not the explanation.

Nsight Compute hardware counters could not be collected because the installed Nsight
Compute reports that it is incompatible with the current CUDA driver. The XProf kernel
durations are CUPTI measurements, while the temporary size below comes directly from XLA's
buffer assignment rather than a traffic estimate.

## IR evidence

After batching and legalization, StableHLO contains one width-8 forward derivative. Its
two tangent inputs are concatenations of the eight original directions. After Enzyme, the
important part of the simplified graph is:

```mlir
%du = stablehlo.concatenate %du0, ..., %du7, dim = 0
%dv = stablehlo.concatenate %dv0, ..., %dv7, dim = 0

%twodu = stablehlo.add %du, %du
%a = stablehlo.multiply %broadcast_v, %twodu
%b = stablehlo.multiply %dv, %broadcast_u_squared
%nonlinear_tangent = stablehlo.add %a, %b

%out_u = stablehlo.add %u_linear_part, %nonlinear_tangent
%out_v = stablehlo.subtract %v_linear_part, %nonlinear_tangent

%u0 = stablehlo.slice %out_u [0:1, ...]
// ... the remaining same-partition slices of out_u and out_v ...
```

In the captured N=3 IR these are `%8`/`%17`, `%319`, `%320`, and `%326`; `%319` is the
multi-use `tensor<8x3x3xf64>` value. The production N=4096 XLA HLO preserves the same
structure:

- `loop_add_fusion` returns `f64[8,4096,4096]` and computes only the shared nonlinear
  tangent.
- `loop_concatenate_fusion` takes that tensor as operand 0 and uses it twice, once in an
  add and once in a subtract.
- The thunk sequence executes the two kernels serially.
- XLA buffer assignment reserves 1,073,741,824 bytes for the intermediate. Total assigned
  memory rises from 4.25 GiB unbatched to 5.25 GiB batched.

The unbatched HLO has only `loop_concatenate_fusion`. Its lane-local nonlinear expressions
remain inside that output fusion, so no derivative-sized intermediate crosses a kernel
boundary.

## Causal ablation

A diagnostic source variant wrote the two mathematically equivalent primal nonlinear terms
with different association:

```julia
nonlinear_u = (u .* u) .* v
nonlinear_v = u .* (u .* v)
```

The N=16 primal outputs differ from the normal source by at most `2.84e-14`. The different
expression trees prevent XLA from forming the shared width-8 nonlinear tangent. With
batching still enabled and every other benchmark setting unchanged:

| Measurement | Normal batched | Distinct-expression ablation |
|---|---:|---:|
| Host median | 9.357 ms | 8.111 ms |
| Summed kernel time | 8.448 ms | 7.791 ms |
| Largest non-output temporary | 1.00 GiB | 128 MiB |

The ablation still has a small 128 MiB `wrapped_multiply` temporary, but eliminates the
1.00 GiB width-8 temporary and recovers 13.3% of synchronized runtime. This is a diagnostic
confirmation, not a suggested change to the Brusselator source.

## Compiler fix suggested by the profile

The relevant StableHLO pattern is the same-partition
`Slice(Elementwise(Concat(...)))` graph discussed previously. The cleanup should push each
terminal lane slice backward through the elementwise DAG and simplify a slice of the lane
concatenation to the corresponding input. For this graph it must also clone the sliced form
of `%nonlinear_tangent` into its two consumers instead of preserving one materialized wide
common subexpression.

The rewrite should be profitability-driven rather than unconditional for arbitrary shared
producers. Here the decision is unambiguous: materializing saves 1.865 ms in the consumer
but costs 3.370 ms in the producer, so recomputing/fusing is 1.506 ms cheaper. Useful guards
are:

- every slice set forms the same complete, non-overlapping partition;
- concatenated operands and slices have matching rank and partition boundaries;
- the traversed operations are side-effect-free elementwise operations;
- the temporary is large relative to the small amount of cloned scalar arithmetic.

After such a cleanup, the expected verification is one terminal GPU fusion, no
`f64[K,N,N]` allocation in XLA buffer assignment, and batched runtime no worse than the
unbatched kernel for this workload.
