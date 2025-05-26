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
