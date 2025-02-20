# [Getting Started](@id getting-started)

## Installation

Install [Julia v1.10 or above](https://julialang.org/downloads/). Reactant.jl is available
through the Julia package manager. You can enter it by pressing `]` in the REPL and then
typing `add Reactant`. Alternatively, you can also do

```julia
import Pkg
Pkg.add("Reactant")
```

## Quick Start

Reactant provides two new array types at its core, a ConcretePJRTArray and a TracedRArray. A
ConcretePJRTArray is an underlying buffer to whatever device data you wish to store and can be
created by converting from a regular Julia Array.

```@example quickstart
using Reactant

julia_data = ones(2, 10)
reactant_data = Reactant.ConcretePJRTArray(julia_data)
```

You can also create a ConcretePJRTArray-version of an arbitrary data type by tracing through
the structure, like below.

```@example quickstart
struct Pair{A,B}
   x::A
   y::B
end

pair = Pair(ones(3), ones(10))

reactant_pair = Reactant.to_rarray(pair)
```

To compile programs using ConcretePJRTArray's, one uses the compile function, like as follows:

```@example quickstart
input1 = Reactant.ConcretePJRTArray(ones(10))
input2 = Reactant.ConcretePJRTArray(ones(10))

function sinsum_add(x, y)
   return sum(sin.(x) .+ y)
end

f = @compile sinsum_add(input1,input2)

# one can now run the program
f(input1, input2)
```


## Tips

### Empty Cache

When you encounter OOM (Out of Memory) errors, you can try to clear the cache by using Julia's builtin `GC.gc()` between memory-intensive operations.

!!! note
    This will only free memory which is not currently live. If the result of compiled function was stored in a vector, it will still be alive and `GC.gc()` won't free it.

```julia
using Reactant
n = 500_000_000
input1 = Reactant.ConcretePJRTArray(ones(n))
input2 = Reactant.ConcretePJRTArray(ones(n))

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
 [5] (::Reactant.Compiler.Thunk{…})(::ConcretePJRTArray{…}, ::ConcretePJRTArray{…})
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





