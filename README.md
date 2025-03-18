# Reactant.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://enzymead.github.io/Reactant.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzymead.github.io/Reactant.jl/dev)
[![Build Status](https://github.com/EnzymeAD/Reactant.jl/workflows/CI/badge.svg)](https://github.com/EnzymeAD/Reactant.jl/actions)
[![Coverage](https://codecov.io/gh/EnzymeAD/Reactant.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EnzymeAD/Reactant.jl)

> [!WARNING]  
> This package is under active development at the moment and may change its API and supported end systems at any time. End-users are advised to wait until a corresponding release with broader availability is made. Package developers are suggested to try out Reactant for integration, but should be advised of the same risks.

Reactant takes Julia function and compile it into MLIR and run fancy optimizations on top of it, including using EnzymeMLIR for automatic differentiation, and create relevant executables for CPU/GPU/TPU via XLA. It presently operates as a tracing system. Compiled functions will assume the same control flow pattern as was originally taken by objects used at compile time, and control flow (e.g. if, for) as well as any type instabilities will be removed. The benefits of this approach is immediately making all such code available for advanced optimization with little developer effort. This system and corresponding semantics is subject to change to a (potentially partial) source rewriter in the future.

Reactant provides two new array types at its core, a ConcreteRArray and a TracedRArray. A ConcreteRArray is an underlying buffer to whatever device data you wish to store and can be created by converting from a regular Julia Array.

```julia
using Reactant

julia_data = ones(2, 10)
reactant_data = Reactant.ConcreteRArray(julia_data)
```

You can also create a ConcreteRArray-version of an arbitrary data type by tracing through the structure, like below. This method will automatically handle recursive data structures or shared objects.

```julia
struct Pair{A,B}
   x::A
   y::B
end

pair = Pair(ones(3), ones(10))

reactant_pair = Reactant.to_rarray(pair)
``` 

To compile programs using ConcreteRArray's, one uses the compile function, like as follows:

```julia
input1 = Reactant.ConcreteRArray(ones(10))
input2 = Reactant.ConcreteRArray(ones(10))

function sinsum_add(x, y)
   return sum(sin.(x) .+ y)
end

f = @compile sinsum_add(input1,input2)

# one can now run the program
f(input1, input2)
```

The generated function can be called with data structures which match the same types (and sizes) as were originally compiled with. Reactant (presently, see above) only captures updates to `ConcreteRArray` and as such any updates to other data (such as a regular int counter) will not be reflected in generated compiled functions. Similarly if there are any branches or conditional behavior based on any non-ConcreteRArray data, they will not be reflected in the generated function.

When being compiled, functions will not have access to the actual values of data within ConcreteRArrays, and will instead be passed TracedRArrays to reflect this semantic distinction.

One can automatically leverage Reactant to run programs on accelerators, by specifying the desired device client. For example, to run programs on GPU one can initialize the default device as follows (or alternatively construct RArrays with a device).

```julia
using Reactant
Reactant.set_default_backend("gpu")

# ones favorite code will now all be executed on GPU, no CUDA.jl dependency even required!
```
