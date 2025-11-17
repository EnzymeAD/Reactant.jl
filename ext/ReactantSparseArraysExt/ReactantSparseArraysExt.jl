module ReactantSparseArraysExt

using Reactant: Reactant, TracedRNumber
using SparseArrays:
    SparseArrays, ReadOnly, AbstractSparseArray, CHOLMOD, AbstractSparseMatrixCSC

include("Errors.jl")
include("ReadOnly.jl")

end
