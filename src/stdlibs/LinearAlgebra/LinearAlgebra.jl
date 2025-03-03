module TracedLinearAlgebra

using ..Reactant:
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    AnyTracedRMatrix,
    AnyTracedRVector,
    AnyTracedRVecOrMat,
    WrappedTracedRArray,
    unwrapped_eltype,
    Ops,
    MLIR
using ReactantCore: @trace
using GPUArraysCore: @allowscalar

using ..TracedUtils: TracedUtils, get_mlir_data, materialize_traced_array, set_mlir_data!

using LinearAlgebra
using LinearAlgebra: BLAS, LAPACK

include("Core.jl")
include("LAPACK.jl")
include("WrappedArrays.jl")
include("Utils.jl")

end
