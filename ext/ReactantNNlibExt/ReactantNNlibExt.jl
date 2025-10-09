module ReactantNNlibExt

using Reactant:
    Reactant, Ops, TracedRArray, AnyTracedRArray, TracedRNumber, @reactant_overlay
using Reactant.TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!
using Reactant.Ops: @opcall
using ReactantCore: materialize_traced_array, @trace

using LinearAlgebra: LinearAlgebra
using NNlib: NNlib, DenseConvDims
using GPUArraysCore: @allowscalar
using Statistics: mean

include("Overlay.jl")
include("Ops.jl")
include("Implementations.jl")

end # module ReactantNNlibExt
