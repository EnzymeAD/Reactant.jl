# We don't yet support sparse arrays. Throwing errors on this dispatches to avoid
# ambiguities.
function Base.getindex(::AbstractSparseArray{TracedRNumber{T},N,1}, ::Int64) where {T,N}
    return error("Sparse arrays are not supported by reactant yet.")
end

function Base.getindex(
    ::AbstractSparseMatrixCSC{TracedRNumber{T}}, ::Int64, ::Int64
) where {T}
    return error("Sparse arrays are not supported by reactant yet.")
end

function Base.getindex(::CHOLMOD.Sparse{TracedRNumber{T}}, ::Int64, ::Int64) where {T}
    return error("Sparse arrays are not supported by reactant yet.")
end
