function Base.getindex(A::ReadOnly{<:Reactant.TracedRNumber,1}, inds::AbstractArray)
    return getindex(parent(A), inds)
end

function Base.getindex(A::ReadOnly{<:Reactant.TracedRNumber,1}, ::Colon)
    return getindex(parent(A), :)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber},
    i::Union{Integer,Reactant.TracedRNumber{<:Integer}},
)
    return getindex(parent(A), i)
end

function Base.getindex(A::ReadOnly{<:Reactant.TracedRNumber}, inds::AbstractArray)
    return getindex(parent(A), inds)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber,N}, idx::CartesianIndex{N}
) where {N}
    return getindex(parent(A), idx)
end

function Base.getindex(A::ReadOnly{<:Reactant.TracedRNumber,1}, idx::CartesianIndex{1})
    return getindex(parent(A), idx)
end

function Base.getindex(A::ReadOnly{<:Reactant.TracedRNumber}, ::Colon)
    return getindex(parent(A), :)
end

function Base.getindex(
    x::ReadOnly{<:TracedRNumber,N},
    idx::Vararg{Union{Integer,Reactant.TracedRNumber{<:Integer}},N},
) where {N}
    return getindex(parent(x), idx...)
end

# Disambiguates against the generic traced-array `getindex`.
function Base.getindex(
    x::ReadOnly{<:TracedRNumber{T},N}, idx::Vararg{Union{Int,Reactant.TracedRNumber{Int}},N}
) where {T,N}
    return getindex(parent(x), idx...)
end

function Base.getindex(
    x::SparseArrays.ReadOnly{<:Reactant.TracedRNumber,1},
    idx::Union{Int64,Reactant.TracedRNumber{Int64}},
)
    return getindex(parent(x), idx)
end

function Base.getindex(
    x::SparseArrays.ReadOnly{<:Reactant.TracedRNumber,N},
    idx::Vararg{Union{Int64,Reactant.TracedRNumber{Int64}},N},
) where {N}
    return getindex(parent(x), idx...)
end
