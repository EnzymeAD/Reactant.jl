# Why differentiation batching slows the compressed Brusselator Jacobian

Date: 2026-07-14

## Finding

At K=8, differentiation batching turns one fused GPU kernel into two kernels and
materializes a 1.00 GiB `f64[8,4096,4096]` temporary. The temporary is the tangent of the
shared nonlinear term `u^2*v`; it is consumed once with a plus sign by the first species
and once with a minus sign by the second species. XLA chooses to compute that multi-use
wide value once and store it rather than clone it into the terminal output fusion.

This is the dominant identified structural cause of the measured slowdown. In the matched
kernel accounting, precomputing the shared term makes the final kernel 1.865 ms faster, but
the producer kernel costs 3.370 ms, leaving a measured 1.506 ms net regression. The result
is not a kernel-launch-overhead problem: both kernels process hundreds of millions of
values, and the extra kernel accounts for 39.9% of batched device time.

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

The complete regression input is
[`regressions/brusselator-k8-n4096-batched.mlir`](regressions/brusselator-k8-n4096-batched.mlir).
It is the unmodified StableHLO module sent to XLA by the profiled batched run, rather than
a minimized or hand-written testcase. The following lines are verbatim excerpts from it.
The companion
[`regressions/brusselator-slice-elementwise-concat-min.mlir`](regressions/brusselator-slice-elementwise-concat-min.mlir)
keeps the exact `8x4096x4096` production types and eight one-lane partitions while reducing
the graph to the missed optimization. Running it through `--enzyme-hlo-opt` still leaves
the two concatenations, the shared multiply and its add/subtract consumers at full width,
and all sixteen slices.

The eight tangent inputs first become the leading dimension of two `8x4096x4096` values:

```mlir
%8 = stablehlo.concatenate %0, %1, %2, %3, %4, %5, %6, %7, dim = 0 : (tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
%17 = stablehlo.concatenate %9, %10, %11, %12, %13, %14, %15, %16, dim = 0 : (tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
```

The shared nonlinear tangent is `%1033`. It feeds both species before the results are
sliced back into the original eight partitions:

```mlir
%1026 = stablehlo.multiply %8, %1025 : tensor<8x4096x4096xf64> loc(#loc)
%1027 = stablehlo.add %1026, %1026 : tensor<8x4096x4096xf64> loc(#loc)
%1028 = stablehlo.multiply %arg1, %arg1 : tensor<4096x4096xf64> loc(#loc)
%1029 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
%1030 = stablehlo.multiply %1027, %1029 : tensor<8x4096x4096xf64> loc(#loc)
%1031 = stablehlo.broadcast_in_dim %1028, dims = [1, 2] : (tensor<4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
%1032 = stablehlo.multiply %17, %1031 : tensor<8x4096x4096xf64> loc(#loc)
%1033 = stablehlo.add %1030, %1032 : tensor<8x4096x4096xf64> loc(#loc)
%1034 = stablehlo.add %1024, %1033 : tensor<8x4096x4096xf64> loc(#loc)
%1035 = stablehlo.broadcast_in_dim %cst_1, dims = [1, 2] : (tensor<4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
%1036 = stablehlo.multiply %8, %1035 : tensor<8x4096x4096xf64> loc(#loc)
%1037 = stablehlo.add %1034, %1036 : tensor<8x4096x4096xf64> loc(#loc)
%1038 = stablehlo.multiply %1022, %1023 : tensor<8x4096x4096xf64> loc(#loc)
%1039 = stablehlo.broadcast_in_dim %cst_0, dims = [1, 2] : (tensor<4096x4096xf64>) -> tensor<8x4096x4096xf64> loc(#loc)
%1040 = stablehlo.multiply %8, %1039 : tensor<8x4096x4096xf64> loc(#loc)
%1041 = stablehlo.add %1038, %1040 : tensor<8x4096x4096xf64> loc(#loc)
%1042 = stablehlo.subtract %1041, %1033 : tensor<8x4096x4096xf64> loc(#loc)
%1043 = stablehlo.slice %1037 [0:1, 0:4096, 0:4096] : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64> loc(#loc)
%1045 = stablehlo.slice %1037 [1:2, 0:4096, 0:4096] : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64> loc(#loc)
%1059 = stablehlo.slice %1042 [0:1, 0:4096, 0:4096] : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64> loc(#loc)
%1061 = stablehlo.slice %1042 [1:2, 0:4096, 0:4096] : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64> loc(#loc)
```

XLA preserves the multi-use width-8 value and makes the dataflow across the kernel boundary
explicit:

```text
%loop_add_fusion = f64[8,4096,4096]{2,1,0} fusion(%Arg_2.1, %Arg_1.1, %Arg_18.1, %Arg_16.1, %Arg_14.1, /*index=5*/%Arg_12.1, %Arg_10.1, %Arg_8.1, %Arg_6.1, %Arg_4.1, /*index=10*/%Arg_17.1, %Arg_15.1, %Arg_13.1, %Arg_11.1, %Arg_9.1, /*index=15*/%Arg_7.1, %Arg_5.1, %Arg_3.1), kind=kLoop, calls=%fused_add, backend_config={"operation_queue_id":"0","force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID","native_emitter_backend_config":{"type":"NATIVE_EMITTER_TYPE_INVALID","unroll_factor":0}}
ROOT %loop_concatenate_fusion = f64[8,33554432]{1,0} fusion(%loop_add_fusion, %Arg_17.1, %Arg_15.1, %Arg_13.1, %Arg_11.1, /*index=5*/%Arg_9.1, %Arg_7.1, %Arg_5.1, %Arg_3.1, %Arg_18.1, /*index=10*/%Arg_16.1, %Arg_14.1, %Arg_12.1, %Arg_10.1, %Arg_8.1, /*index=15*/%Arg_6.1, %Arg_4.1), kind=kLoop, calls=%fused_concatenate, backend_config={"operation_queue_id":"0","force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID","native_emitter_backend_config":{"type":"NATIVE_EMITTER_TYPE_INVALID","unroll_factor":0}}
```

The corresponding buffer assignment is:

```text
allocation 20: size 1073741824, preallocated-temp:
 value: <319 loop_add_fusion @0> (size=1073741824,offset=0): f64[8,4096,4096]{2,1,0}
```

The thunk sequence executes the two kernels serially. Total assigned memory rises from
4.25 GiB unbatched to 5.25 GiB batched.

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
confirmation, not a suggested change to the Brusselator source. Relative to the unbatched
baseline it recovers about 85% of the host-median gap but about 44% of the summed-kernel-time
gap; changing association is therefore strong causal evidence for the wide temporary, not
proof that this one rewrite explains every residual timing difference.

## Compiler fix suggested by the profile

The relevant StableHLO pattern is the same-partition
`Slice(Elementwise(Concat(...)))` graph discussed previously. The cleanup should push each
terminal lane slice backward through the elementwise DAG and simplify a slice of the lane
concatenation to the corresponding input. For this graph it must also clone the sliced form
of `%nonlinear_tangent` into its two consumers instead of preserving one materialized wide
common subexpression.

The minimized reproducer also exposes why the current `SliceElementwise` rewrite does not
start this cleanup. It computes one bounding slice across every slice user and only rewrites
when that bounding slice is a strict subset of the elementwise result. The eight lanes here
cover `[0:8, 0:4096, 0:4096]`, exactly the full result, so the pattern reports no change. It
therefore never exposes the subsequent `Slice(Concat)` simplifications, even though every
slice uses the same concatenation partition.

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
