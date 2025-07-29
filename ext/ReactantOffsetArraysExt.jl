module ReactantOffsetArraysExt

using OffsetArrays
using OffsetArrays: OffsetArray, OffsetVector
using Reactant: Reactant, MLIR, Ops, TracedRArray, AbstractConcreteArray

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{<:OffsetArray}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    N = ndims(OA)
    T = OffsetArrays.parenttype(OA)
    T2 = Reactant.traced_type_inner(T, seen, mode, track_numbers, sharding, runtime)
    return OffsetArray{eltype(T2),N,T2}
end

# Because why wouldn't offset arrays assume that scalar indexing is the faster way
function Base.getindex(
    x::OffsetArray{T,N,<:AbstractConcreteArray}, args::Vararg{Any,N}
) where {T,N}
    args = [arg isa Colon ? Base.OneTo(size(x, i)) : arg for (i, arg) in enumerate(args)]
    offset_indices = [arg .- x.offsets[i] for (i, arg) in enumerate(args)]
    return getindex(parent(x), offset_indices...)
end
function Base.getindex(
    x::OffsetArray{T,N,<:AbstractConcreteArray},
    args::Vararg{Union{Int,AbstractUnitRange{Int}},N},
) where {T,N}
    offset_indices = [arg .- x.offsets[i] for (i, arg) in enumerate(args)]
    return getindex(parent(x), offset_indices...)
end

function Base.getindex(x::OffsetVector{T,<:AbstractConcreteArray}, index::Int) where {T}
    return getindex(parent(x), index - x.offsets[1])
end
function Base.getindex(
    x::OffsetVector{T,<:AbstractConcreteArray}, indices::AbstractUnitRange{Int}
) where {T}
    offset_indices = indices .- x.offsets[1]
    return getindex(parent(x), offset_indices)
end

function Base.getindex(
    x::OffsetVector{Reactant.TracedRNumber{T}, Reactant.TracedRArray{T, 1}}, indices::Base.OneTo{Int}
) where {T}
    offset_indices = indices .- x.offsets[1]
    return getindex(parent(x), offset_indices)
end

parentindex(r::OffsetArrays.IdOffsetRange, i) = i .- r.offset
function Base.getindex(
    a::OffsetArray{<:Reactant.TracedRNumber,N}, indices::Vararg{Union{Int,AbstractArray},N}
) where {N}
    J = map(parentindex, axes(a), indices)
    return parent(a)[J...]
end

function Base.getindex(a::OffsetVector{<:Reactant.TracedRNumber}, indices::Int)
    J = parentindex(Base.axes1(a), indices)
    return parent(a)[J]
end

end
