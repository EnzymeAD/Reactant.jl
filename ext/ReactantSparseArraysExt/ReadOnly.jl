function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},1,V}, inds::AbstractArray
) where {T,V<:AbstractArray{<:Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), inds)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},1,V}, ::Colon
) where {T,V<:AbstractArray{<:Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), :)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},N,V}, inds::AbstractArray
) where {T,N,V<:AbstractArray{<:Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), inds)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},N,V}, idx::CartesianIndex{N}
) where {T,N,V<:AbstractArray{<:Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), idx)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},1,V}, idx::CartesianIndex{1}
) where {T,V<:AbstractArray{<:Reactant.TracedRNumber{T},1}}
    return getindex(parent(A), idx)
end

function Base.getindex(
    A::ReadOnly{<:Reactant.TracedRNumber{T},N,V}, ::Colon
) where {T,N,V<:AbstractArray{<:Reactant.TracedRNumber{T},N}}
    return getindex(parent(A), :)
end

# ReadOnly's own `getindex` just forwards to the parent, but it ties with the
# traced-array `getindex` methods; these covers forward the same way and are
# shaped per traced element type with invariant storage, exactly like the
# traced-array methods they disambiguate against.
for ET in (
    Reactant.TracedRInteger,
    Reactant.TracedRFloat,
    Reactant.TracedRComplex,
    Reactant.TracedRReal,
    Reactant.TracedRNumber,
)
    @eval begin
        function Base.getindex(
            x::ReadOnly{$ET{T},N,V},
            idx::Vararg{Union{Integer,Reactant.TracedRNumber{<:Integer}},N},
        ) where {T,N,V<:AbstractArray{$ET{T},N}}
            return getindex(parent(x), idx...)
        end
        function Base.getindex(
            x::ReadOnly{$ET{T},N,V}, idx::Union{Integer,Reactant.TracedRNumber{<:Integer}}
        ) where {T,N,V<:AbstractArray{$ET{T},N}}
            return getindex(parent(x), idx)
        end
        function Base.getindex(
            x::ReadOnly{$ET{T},N,V}, idx::Vararg{Union{Int,Reactant.TracedRNumber{Int}},N}
        ) where {T,N,V<:AbstractArray{$ET{T},N}}
            return getindex(parent(x), idx...)
        end
        function Base.getindex(
            x::ReadOnly{$ET{T},1,V}, idx::Union{Int,Reactant.TracedRNumber{Int}}
        ) where {T,V<:AbstractVector{$ET{T}}}
            return getindex(parent(x), idx)
        end
    end
end
