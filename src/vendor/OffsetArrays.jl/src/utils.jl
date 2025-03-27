### Low-level utilities ###

_indexoffset(r::AbstractRange) = first(r) - 1
_indexoffset(i::Integer) = 0
_indexlength(r::AbstractRange) = length(r)
_indexlength(i::Integer) = Int(i)
_indexlength(i::Colon) = Colon()

# utility methods used in reshape
# we don't use _indexlength in this to avoid converting the arguments to Int
_checksize(ind::Integer, s) = ind == s
_checksize(ind::AbstractUnitRange, s) = length(ind) == s

_toaxis(i::Integer) = Base.OneTo(i)
_toaxis(i) = i

_strip_IdOffsetRange(r::IdOffsetRange) = parent(r)
_strip_IdOffsetRange(r) = r

_offset(axparent::AbstractUnitRange, ax::AbstractUnitRange) = first(ax) - first(axparent)
_offset(axparent::AbstractUnitRange, ::Union{Integer, Colon}) = 1 - first(axparent)

_offsets(A::AbstractArray) = map(ax -> first(ax) - 1, axes(A))
_offsets(A::AbstractArray, B::AbstractArray) = map(_offset, axes(B), axes(A))

"""
    OffsetArrays.AxisConversionStyle(typeof(indices))

`AxisConversionStyle` declares if `indices` should be converted to a single `AbstractUnitRange{Int}`
or to a `Tuple{Vararg{AbstractUnitRange{Int}}}` while flattening custom types into indices.
This method is called after `to_indices(A::Array, axes(A), indices)` to provide
further information in case `to_indices` does not return a `Tuple` of `AbstractUnitRange{Int}`.

Custom index types should extend `AxisConversionStyle` and return either `OffsetArray.SingleRange()`,
which is the default, or `OffsetArray.TupleOfRanges()`. In the former case, the type `T` should
define `Base.convert(::Type{AbstractUnitRange{Int}}, ::T)`, whereas in the latter it should define
`Base.convert(::Type{Tuple{Vararg{AbstractUnitRange{Int}}}}, ::T)`.

An example of the latter is `CartesianIndices`, which is converted to a `Tuple` of
`AbstractUnitRange{Int}` while flattening the indices.

# Example
```jldoctest; setup=:(using OffsetArrays)
julia> struct NTupleOfUnitRanges{N}
           x ::NTuple{N, UnitRange{Int}}
       end

julia> Base.to_indices(A, inds, t::Tuple{NTupleOfUnitRanges{N}}) where {N} = t;

julia> OffsetArrays.AxisConversionStyle(::Type{NTupleOfUnitRanges{N}}) where {N} = OffsetArrays.TupleOfRanges();

julia> Base.convert(::Type{Tuple{Vararg{AbstractUnitRange{Int}}}}, t::NTupleOfUnitRanges) = t.x;

julia> a = zeros(3, 3);

julia> inds = NTupleOfUnitRanges((3:5, 2:4));

julia> oa = OffsetArray(a, inds);

julia> axes(oa, 1) == 3:5
true

julia> axes(oa, 2) == 2:4
true
```
"""
abstract type AxisConversionStyle end
struct SingleRange <: AxisConversionStyle end
struct TupleOfRanges <: AxisConversionStyle end

AxisConversionStyle(::Type) = SingleRange()
AxisConversionStyle(::Type{<:CartesianIndices}) = TupleOfRanges()

_convertTupleAbstractUnitRange(x) = _convertTupleAbstractUnitRange(AxisConversionStyle(typeof(x)), x)
_convertTupleAbstractUnitRange(::SingleRange, x) = (convert(AbstractUnitRange{Int}, x),)
_convertTupleAbstractUnitRange(::TupleOfRanges, x) = convert(Tuple{Vararg{AbstractUnitRange{Int}}}, x)

_toAbstractUnitRanges(t::Tuple) = (_convertTupleAbstractUnitRange(first(t))..., _toAbstractUnitRanges(tail(t))...)
_toAbstractUnitRanges(::Tuple{}) = ()

# ensure that the indices are consistent in the constructor
_checkindices(A::AbstractArray, indices, label) = _checkindices(ndims(A), indices, label)
function _checkindices(N::Integer, indices, label)
    throw_argumenterror(N, indices, label) = throw(ArgumentError(label*" $indices are not compatible with a $(N)D array"))
    N == length(indices) || throw_argumenterror(N, indices, label)
end

@inline _indexedby(r::AbstractVector, ax::Tuple{Any}) = _indexedby(r, ax[1])
@inline _indexedby(r::AbstractUnitRange{<:Integer}, ::Base.OneTo) = no_offset_view(r)
@inline _indexedby(r::AbstractUnitRange{Bool}, ::Base.OneTo) = no_offset_view(r)
@inline _indexedby(r::AbstractVector, ::Base.OneTo) = no_offset_view(r)
@inline function _indexedby(r::AbstractUnitRange{<:Integer}, ax::AbstractUnitRange)
	of = convert(eltype(r), first(ax) - 1)
	IdOffsetRange(_subtractoffset(r, of), of)
end
@inline _indexedby(r::AbstractUnitRange{Bool}, ax::AbstractUnitRange) = OffsetArray(r, ax)
@inline _indexedby(r::AbstractVector, ax::AbstractUnitRange) = OffsetArray(r, ax)

# These functions are equivalent to the broadcasted operation r .- of
# However these ensure that the result is an AbstractRange even if a specific
# broadcasting behavior is not defined for a custom type
@inline _subtractoffset(r::AbstractUnitRange, of) = UnitRange(first(r) - of, last(r) - of)
@inline _subtractoffset(r::AbstractRange, of) = range(first(r) - of, stop = last(r) - of, step = step(r))

# similar to _subtractoffset, except these evaluate r .+ of
@inline _addoffset(r::AbstractUnitRange, of) = UnitRange(first(r) + of, last(r) + of)
@inline _addoffset(r::AbstractRange, of) = range(first(r) + of, stop = last(r) + of, step = step(r))

if VERSION <= v"1.7.0-DEV.1039"
    _contiguousindexingtype(r::AbstractUnitRange{<:Integer}) = UnitRange{Int}(r)
else
    _contiguousindexingtype(r::AbstractUnitRange{<:Integer}) = r
end

_of_eltype(::Type{T}, M::AbstractArray{T}) where {T} = M
_of_eltype(T, M::AbstractArray) = map(T, M)

# filter the arguments to reshape to check if there are any ranges
# If not, we may pop the parent array
_filterreshapeinds(t::Tuple{AbstractUnitRange, Vararg{Any}}) = t
_filterreshapeinds(t::Tuple) = _filterreshapeinds(tail(t))
_filterreshapeinds(t::Tuple{}) = t
_popreshape(A::AbstractArray, ax::Tuple{Vararg{Base.OneTo}}, inds::Tuple{}) = no_offset_view(A)
_popreshape(A::AbstractArray, ax, inds) = A
