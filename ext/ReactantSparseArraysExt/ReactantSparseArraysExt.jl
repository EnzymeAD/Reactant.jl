module ReactantSparseArraysExt

using Reactant: Reactant, TracedRInteger, TracedRNumber
using SparseArrays:
    SparseArrays,
    ReadOnly,
    AbstractSparseArray,
    AbstractSparseVector,
    CHOLMOD,
    AbstractSparseMatrixCSC

include("Errors.jl")
include("ReadOnly.jl")

# Scalar indexing disambiguators against the generic traced-array `getindex`;
# they delegate to the SparseArrays implementations.
function Base.getindex(
    x::AbstractSparseMatrixCSC{<:TracedRNumber},
    i::Union{Int,TracedRInteger{Int}},
    j::Union{Int,TracedRInteger{Int}},
)
    return Base.invoke(getindex, Tuple{AbstractSparseMatrixCSC,Integer,Integer}, x, i, j)
end

function Base.getindex(
    x::AbstractSparseVector{<:TracedRNumber}, i::Union{Int,TracedRInteger{Int}}
)
    return Base.invoke(getindex, Tuple{AbstractSparseVector,Integer}, x, i)
end

function Base.getindex(
    x::CHOLMOD.Sparse{<:TracedRNumber},
    i::Union{Int,TracedRInteger{Int}},
    j::Union{Int,TracedRInteger{Int}},
)
    return Base.invoke(getindex, Tuple{CHOLMOD.Sparse,Integer,Integer}, x, i, j)
end

end
