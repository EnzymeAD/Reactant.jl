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

Reactant provides two new array types at its core, a ConcreteRArray and a TracedRArray. A
ConcreteRArray is an underlying buffer to whatever device data you wish to store and can be
created by converting from a regular Julia Array.

```@example quickstart
using Reactant

julia_data = ones(2, 10)
reactant_data = Reactant.ConcreteRArray(julia_data)
```

You can also create a ConcreteRArray-version of an arbitrary data type by tracing through
the structure, like below.

```@example quickstart
struct Pair{A,B}
   x::A
   y::B
end

pair = Pair(ones(3), ones(10))

reactant_pair = Reactant.to_rarray(pair)
```

To compile programs using ConcreteRArray's, one uses the compile function, like as follows:

```@example quickstart
input1 = Reactant.ConcreteRArray(ones(10))
input2 = Reactant.ConcreteRArray(ones(10))

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

Note：This only will free memory which is not currently live. If the result of compiled function was stored in a vector, it would still be live and `GC.gc()` would not free it.

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
