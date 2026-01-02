# FAQs

## XLA auto-tuner: Results do not match the reference. This is likely a bug/unexpected loss of precision

If you see this error with the CUDA backend, use a scoped value to increase the precision
of the dot-general algorithm.

```julia
Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGH) do
    @compile ...
end
```

For more information, see [this XLA issue](https://github.com/openxla/xla/issues/23934).

## Emptying the cache to avoid OOM issues

When you encounter OOM (Out of Memory) errors, you can try to clear the cache by using
Julia's builtin `GC.gc()` between memory-intensive operations.

!!! note

    This will only free memory which is not currently live. If the result of compiled
    function was stored in a vector, it will still be alive and `GC.gc()` won't free it.

```julia
using Reactant
n = 500_000_000
input1 = Reactant.ConcreteRArray(ones(n))
input2 = Reactant.ConcreteRArray(ones(n))

function sin_add(x, y)
   return sin.(x) .+ y
end

f = @compile sin_add(input1,input2)

for i = 1:10
   GC.gc()
   @info "gc... $i"
   f(input1, input2) # May cause OOM here for a 24GB GPU if GC is not used
end
```

If you **don't** use `GC.gc()` here, this may cause an OOM:

```bash
[ Info: gc... 1
[ Info: gc... 2
[ Info: gc... 3
...
E0105 09:48:28.755177  110350 pjrt_stream_executor_client.cc:3088] Execution of replica 0 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4000000000 bytes.
ERROR: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4000000000 bytes.

Stacktrace:
 [1] reactant_err(msg::Cstring)
   @ Reactant.XLA ~/.julia/packages/Reactant/7m11i/src/XLA.jl:104
 [2] macro expansion
   @ ~/.julia/packages/Reactant/7m11i/src/XLA.jl:357 [inlined]
 [3] ExecutableCall
   @ ~/.julia/packages/Reactant/7m11i/src/XLA.jl:334 [inlined]
 [4] macro expansion
   @ ~/.julia/packages/Reactant/7m11i/src/Compiler.jl:798 [inlined]
 [5] (::Reactant.Compiler.Thunk{…})(::ConcreteRArray{…}, ::ConcreteRArray{…})
   @ Reactant.Compiler ~/.julia/packages/Reactant/7m11i/src/Compiler.jl:909
 [6] top-level scope
   @ ./REPL[7]:4
Some type information was truncated. Use `show(err)` to see complete types.
```

After using Julia's built-in `GC.gc()`:

```bash
[ Info: gc... 1
[ Info: gc... 2
[ Info: gc... 3
[ Info: gc... 4
[ Info: gc... 5
[ Info: gc... 6
[ Info: gc... 7
[ Info: gc... 8
[ Info: gc... 9
[ Info: gc... 10
```

## Benchmark results feel suspiciously fast

If you see benchmark results that are suspiciously fast, it's likely because the benchmark
was executed with compiled functions where `sync=false` was used (the default). In this case, the compiled
function will be executed asynchronously, and the benchmark results will be the time it takes
for the function to schedule the computation on the device. Compile functions with `sync=true` to get the actual runtime. You can also use the `Reactant.synchronize` on the result of the computation to block until the computation is complete.

### Example

```julia
using Reactant, BenchmarkTools

function myfunc(x, y)
    return x .+ y
end

x = Reactant.to_rarray(rand(Float32, 1000))
y = Reactant.to_rarray(rand(Float32, 1000))
```

```julia
@benchmark f($x, $y) setup=(f = @compile sync=false myfunc($x, $y))
```

```julia
BenchmarkTools.Trial: 199 samples with 9 evaluations per sample.
 Range (min … max):  2.926 μs … 14.333 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     3.607 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.210 μs ±  1.968 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   █▅▄▂
  ▇█████▇█▅▅▃▃▂▃▄▂▂▁▁▂▁▁▂▁▁▁▁▁▂▁▁▁▁▂▁▁▁▃▁▁▁▁▁▂▁▁▁▁▁▁▁▂▃▁▂▁▁▃ ▂
  2.93 μs        Histogram: frequency by time        12.5 μs <

 Memory estimate: 400 bytes, allocs estimate: 14.
```

```julia
@benchmark f($x, $y) setup=(f = @compile sync=true myfunc($x, $y))
```

```julia
BenchmarkTools.Trial: 221 samples with 8 evaluations per sample.
 Range (min … max):   8.974 μs … 42.443 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     11.688 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.264 μs ±  3.070 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    █▃ ▅▇██▃▄ ▄█▃▄▂▅ ▇ ▂    ▂
  ▇▆████████████████▇███▇▃▆▆██▅▅▅▆▆▅▇▁▆▁▁▃▁▃▁▆▁▁▃▁▁▁▁▃▁▁▁▅▁▁▅ ▅
  8.97 μs         Histogram: frequency by time        20.2 μs <

 Memory estimate: 400 bytes, allocs estimate: 14.
```

```julia
@benchmark begin
    result = f($x, $y);
    Reactant.synchronize(result)
end setup=(f = @compile sync=false myfunc(x, y))
```

```julia
BenchmarkTools.Trial: 233 samples with 8 evaluations per sample.
 Range (min … max):   8.911 μs … 19.609 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     12.479 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.758 μs ±  2.219 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

       ▁▄▄▄▃▂▁   ▃  ▄█▂ ▆▄▂ ▃    ▁  ▃   ▁
  ▄▃▄▆▆████████▆██▆▇███▆███▇█▄▄▆██▇▇█▇▆▄██▇▁▆▄▃▃▁▁▁▁▃▁▁▁▁▄▁▁▃ ▄
  8.91 μs         Histogram: frequency by time        19.3 μs <

 Memory estimate: 400 bytes, allocs estimate: 14.
```

## XLA verbosity flags

XLA has special logging flags that can be used to get more information about the compilation
process. These flags are:

1. `TF_CPP_MAX_VLOG_LEVEL`: This set the max verbosity level for XLA, i.e. all logging
   for `VLOG(level)` where `level <= TF_CPP_MAX_VLOG_LEVEL` will be printed.
2. `TF_CPP_MIN_VLOG_LEVEL`: This set the min verbosity level for XLA, i.e. all logging
   for `VLOG(level)` where `level >= TF_CPP_MIN_VLOG_LEVEL` will be printed.

## Timing Reactant Code

Naively using Julia's builtin profiling mechanisms with `@time`, `@allocated`, etc. will
produce incorrect results. Refer to [our guide on profiling](@ref profiling)
for the correct profiling mechanisms.
