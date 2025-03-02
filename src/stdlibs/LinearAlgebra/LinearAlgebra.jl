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

using ..TracedUtils: TracedUtils, get_mlir_data, materialize_traced_array, set_mlir_data!

using LinearAlgebra

include("Core.jl")
include("WrappedArrays.jl")
include("Utils.jl")

end
