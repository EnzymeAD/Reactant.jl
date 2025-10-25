module ReactantSparseArraysExt

using Reactant: Reactant, TracedRNumber
using SparseArrays:
    SparseArrays, ReadOnly, AbstractSparseArray, CHOLMOD, AbstractSparseMatrixCSC

include("Errors.jl")
include("ReadOnly.jl")

Reactant.use_overlayed_version(::AbstractSparseArray) = false

end
