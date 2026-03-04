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

Reactant provides two new array types at its core, a [`ConcreteRArray`](@ref) and a `TracedRArray`. A
`ConcreteRArray` is an underlying buffer to whatever device data you wish to store and can be
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

!!! note "Tracking scalar numbers"
    By default, `Reactant.to_rarray` only converts arrays — plain Julia numbers are left
    as-is and treated as compile-time constants. If you need scalar values to vary at
    runtime (e.g. a time parameter `t`), use `track_numbers=true`:
    ```julia
    t = Reactant.to_rarray(1.5; track_numbers=true)  # ConcreteRNumber{Float64}
    ```
    See the [Partial Evaluation](@ref partial-evaluation) tutorial for details.

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
