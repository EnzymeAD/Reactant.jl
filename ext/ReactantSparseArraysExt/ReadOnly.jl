function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},1,V}, inds::AbstractArray
) where {T,V<:AbstractArray{Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), inds)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},1,V}, ::Colon
) where {T,V<:AbstractArray{Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), :)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},N,V},
    i::Union{Integer,Reactant.TracedRNumber{<:Integer}},
) where {T,N,V<:AbstractArray{Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), i)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},N,V}, inds::AbstractArray
) where {T,N,V<:AbstractArray{Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), inds)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},N,V}, idx::CartesianIndex{N}
) where {T,N,V<:AbstractArray{Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), idx)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},1,V}, idx::CartesianIndex{1}
) where {T,V<:AbstractArray{Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), idx)
end

function Base.getindex(
    A::ReadOnly{Reactant.TracedRNumber{T},N,V}, ::Colon
) where {T,N,V<:AbstractArray{Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), :)
end

function Base.getindex(
    x::ReadOnly{TracedRNumber{T},N},
    idx::Vararg{Union{Integer,Reactant.TracedRNumber{<:Integer}},N},
) where {T,N}
    return getindex(parent(x), idx...)
end

function Base.getindex(
    x::SparseArrays.ReadOnly{Reactant.TracedRNumber{T},1,V},
    idx::Union{Int64,Reactant.TracedRNumber{Int64}},
) where {T,V<:AbstractArray{Reactant.TracedRNumber{T},1}}
    return getindex(parent(x), idx)
end

function Base.getindex(
    x::SparseArrays.ReadOnly{
        Reactant.TracedRNumber{T},N,V
    } where {V<:AbstractArray{Reactant.TracedRNumber{T},N}},
    idx::Vararg{Union{Int64,Reactant.TracedRNumber{Int64}},N},
) where {T,N}
    return getindex(parent(x), idx...)
end
