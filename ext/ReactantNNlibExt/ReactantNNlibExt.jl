module ReactantNNlibExt

using NNlib
using GPUArraysCore: @allowscalar
using Reactant:
    Reactant, Ops, TracedRArray, AnyTracedRArray, MLIR, TracedRNumber, @reactant_overlay

using Reactant.TracedUtils:
    TracedUtils, materialize_traced_array, get_mlir_data, set_mlir_data!

using ReactantCore: @trace
using LinearAlgebra: LinearAlgebra, triu
using Statistics: mean

include("Overlay.jl")
include("Ops.jl")
include("Implementations.jl")

end # module ReactantNNlibExt
