"""
    ro = IdOffsetRange(r::AbstractUnitRange, offset=0)

Construct an "identity offset range". Numerically, `collect(ro) == collect(r) .+ offset`,
with the additional property that `axes(ro, 1) = axes(r, 1) .+ offset`.
When `r` starts at 1, then `ro[i] == i` and even `ro[ro] == ro`,
i.e., it's the "identity," which is the origin of the "Id" in `IdOffsetRange`.

# Examples

The most common case is shifting a range that starts at 1 (either `1:n` or `Base.OneTo(n)`):
```jldoctest ior
julia> using OffsetArrays: IdOffsetRange

julia> ro = IdOffsetRange(1:3, -2)
IdOffsetRange(values=-1:1, indices=-1:1)

julia> axes(ro, 1)
IdOffsetRange(values=-1:1, indices=-1:1)

julia> ro[-1]
-1

julia> ro[3]
ERROR: BoundsError: attempt to access 3-element $(IdOffsetRange{Int, UnitRange{Int}}) with indices -1:1 at index [3]
```

If the range doesn't start at 1, the values may be different from the indices:
```jldoctest ior
julia> ro = IdOffsetRange(11:13, -2)
IdOffsetRange(values=9:11, indices=-1:1)

julia> axes(ro, 1)     # 11:13 is indexed by 1:3, and the offset is also applied to the axes
IdOffsetRange(values=-1:1, indices=-1:1)

julia> ro[-1]
9

julia> ro[3]
ERROR: BoundsError: attempt to access 3-element $(IdOffsetRange{Int, UnitRange{Int}}) with indices -1:1 at index [3]
```

# Extended help

Construction/coercion preserves the (shifted) values of the input range, but may modify
the indices if required by the specified types. For example,

    r = OffsetArrays.IdOffsetRange{Int,UnitRange{Int}}(3:4)

has `r[1] == 3` and `r[2] == 4`, whereas

    r = OffsetArrays.IdOffsetRange{Int,Base.OneTo{Int}}(3:4)

has `r[3] == 3` and `r[4] == 4`, and `r[1]` would throw a `BoundsError`.
In this latter case, a shift in the axes was needed because `Base.OneTo` ranges
must start with value 1.

!!! warning

    In the future, *conversion* will preserve both the values and
    the indices, throwing an error when this is not achievable. For instance,

        r = convert(OffsetArrays.IdOffsetRange{Int,UnitRange{Int}}, 3:4)

    has `r[1] == 3` and `r[2] == 4` and would satisfy `r == 3:4`, whereas

    ```julia
    julia> convert(OffsetArrays.IdOffsetRange{Int,Base.OneTo{Int}}, 3:4)    # future behavior, not present behavior
    ERROR: ArgumentError: first element must be 1, got 3
    ```

    where the error will arise because the result could not have the same axes as the input.

    An important corollary is that `typeof(r1)(r2)` and `oftype(r1, r2)` will behave differently:
    the first coerces `r2` to be of the type of `r1`, whereas the second converts.
    Developers are urged to future-proof their code by choosing the behavior appropriate for each usage.
"""
struct IdOffsetRange{T<:Integer,I<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
    parent::I
    offset::T

    function IdOffsetRange{T,I}(r::I, offset::T) where {T<:Integer,I<:AbstractUnitRange{T}}
        _bool_check(T, r, offset)
        new{T,I}(r, offset)
    end

    #= This method is necessary to avoid a StackOverflowError in IdOffsetRange{T,I}(r::IdOffsetRange, offset::Integer).
    The type signature in that method is more specific than IdOffsetRange{T,I}(r::I, offset::T),
    so it ends up calling itself if I <: IdOffsetRange.
    =#
    function IdOffsetRange{T,IdOffsetRange{T,I}}(r::IdOffsetRange{T,I}, offset::T) where {T<:Integer,I<:AbstractUnitRange{T}}
        _bool_check(T, r, offset)
        new{T,IdOffsetRange{T,I}}(r, offset)
    end
end

function _bool_check(::Type{Bool}, r, offset)
    # disallow the construction of IdOffsetRange{Bool, UnitRange{Bool}}(true:true, true)
    if offset && (first(r) || last(r))
        throw(ArgumentError("values = $r and offset = $offset can not produce a boolean range"))
    end
    return nothing
end
_bool_check(::Type, r, offset) = nothing

# Construction/coercion from arbitrary AbstractUnitRanges
function IdOffsetRange{T,I}(r::AbstractUnitRange, offset::Integer = 0) where {T<:Integer,I<:AbstractUnitRange{T}}
    rc, o = offset_coerce(I, r)
    return IdOffsetRange{T,I}(rc, convert(T, o+offset)::T)
end
function IdOffsetRange{T}(r::AbstractUnitRange, offset::Integer = 0) where T<:Integer
    rc = convert(AbstractUnitRange{T}, r)::AbstractUnitRange{T}
    return IdOffsetRange{T,typeof(rc)}(rc, convert(T, offset)::T)
end
IdOffsetRange(r::AbstractUnitRange{T}, offset::Integer = 0) where T<:Integer =
    IdOffsetRange{T,typeof(r)}(r, convert(T, offset)::T)

# Coercion from other IdOffsetRanges
IdOffsetRange{T,I}(r::IdOffsetRange{T,I}) where {T<:Integer,I<:AbstractUnitRange{T}} = r
function IdOffsetRange{T,I}(r::IdOffsetRange, offset::Integer = 0) where {T<:Integer,I<:AbstractUnitRange{T}}
    rc, offset_rc = offset_coerce(I, r.parent)
    return IdOffsetRange{T,I}(rc, convert(T, r.offset + offset + offset_rc)::T)
end
IdOffsetRange{T}(r::IdOffsetRange{T}) where {T<:Integer} = r
function IdOffsetRange{T}(r::IdOffsetRange, offset::Integer = 0) where T<:Integer
    return IdOffsetRange{T}(r.parent, r.offset + offset)
end
IdOffsetRange(r::IdOffsetRange) = r

# Constructor to make `show` round-trippable
# try to preserve typeof(values) if the indices are known to be 1-based
_subtractindexoffset(values, indices::Union{Base.OneTo, IdentityUnitRange{<:Base.OneTo}}, offset) = values
_subtractindexoffset(values, indices, offset) = _subtractoffset(values, offset)
function IdOffsetRange(; values::AbstractUnitRange{<:Integer}, indices::AbstractUnitRange{<:Integer})
    length(values) == length(indices) || throw(ArgumentError("values and indices must have the same length"))
    values_nooffset = no_offset_view(values)
    offset = first(indices) - 1
    values_minus_offset = _subtractindexoffset(values_nooffset, indices, offset)
    return IdOffsetRange(values_minus_offset, offset)
end

# Conversions to an AbstractUnitRange{Int} (and to an OrdinalRange{Int,Int} on Julia v"1.6") are necessary
# to evaluate CartesianIndices for BigInt ranges, as their axes are also BigInt ranges
Base.AbstractUnitRange{T}(r::IdOffsetRange) where {T<:Integer} = IdOffsetRange{T}(r)

# https://github.com/JuliaLang/julia/pull/40038
if v"1.6" <= VERSION < v"1.9.0-DEV.642"
    Base.OrdinalRange{T,T}(r::IdOffsetRange) where {T<:Integer} = IdOffsetRange{T}(r)
end

# TODO: uncomment these when Julia is ready
# # Conversion preserves both the values and the indices, throwing an InexactError if this
# # is not possible.
# Base.convert(::Type{IdOffsetRange{T,I}}, r::IdOffsetRange{T,I}) where {T<:Integer,I<:AbstractUnitRange{T}} = r
# Base.convert(::Type{IdOffsetRange{T,I}}, r::IdOffsetRange) where {T<:Integer,I<:AbstractUnitRange{T}} =
#     IdOffsetRange{T,I}(convert(I, r.parent), r.offset)
# Base.convert(::Type{IdOffsetRange{T,I}}, r::AbstractUnitRange) where {T<:Integer,I<:AbstractUnitRange{T}} =
#     IdOffsetRange{T,I}(convert(I, r), 0)

offset_coerce(::Type{Base.OneTo{T}}, r::Base.OneTo) where T<:Integer = convert(Base.OneTo{T}, r), 0
function offset_coerce(::Type{Base.OneTo{T}}, r::AbstractUnitRange) where T<:Integer
    o = first(r) - 1
    return Base.OneTo{T}(last(r) - o), o
end
# function offset_coerce(::Type{Base.OneTo{T}}, r::IdOffsetRange) where T<:Integer
#     rc, o = offset_coerce(Base.OneTo{T}, r.parent)

# Fallback, specialze this method if `convert(I, r)` doesn't do what you need
offset_coerce(::Type{I}, r::AbstractUnitRange) where I<:AbstractUnitRange =
    convert(I, r)::I, 0

@inline Base.parent(r::IdOffsetRange) = r.parent
@inline Base.axes(r::IdOffsetRange) = (axes1(r),)
@inline axes1(r::IdOffsetRange) = IdOffsetRange(Base.axes1(r.parent), r.offset)
if VERSION < v"1.8.2"
    Base.axes1(r::IdOffsetRange) = axes1(r)
end
@inline Base.unsafe_indices(r::IdOffsetRange) = (axes1(r),)
@inline Base.length(r::IdOffsetRange) = length(r.parent)
@inline Base.isempty(r::IdOffsetRange) = isempty(r.parent)
#= We specialize on reduced_indices to work around cases where the parent axis type doesn't
support reduced_index, but the axes do support reduced_indices
The difference is that reduced_index expects the axis type to remain unchanged,
which may not always be possible, eg. for statically sized axes
See https://github.com/JuliaArrays/OffsetArrays.jl/issues/204
=#
function Base.reduced_indices(inds::Tuple{IdOffsetRange, Vararg{IdOffsetRange}}, d::Int)
    parents_reduced = Base.reduced_indices(map(parent, inds), d)
    ntuple(i -> IdOffsetRange(parents_reduced[i], inds[i].offset), Val(length(inds)))
end
Base.reduced_index(i::IdOffsetRange) = typeof(i)(first(i):first(i))
# Workaround for #92 on Julia < 1.4
Base.reduced_index(i::IdentityUnitRange{<:IdOffsetRange}) = typeof(i)(first(i):first(i))
if VERSION < v"1.8.2"
    for f in [:firstindex, :lastindex]
        @eval @inline Base.$f(r::IdOffsetRange) = $f(r.parent) + r.offset
    end
end
for f in [:first, :last]
    # coerce the type to deal with values that get promoted on addition (eg. Bool)
    @eval @inline Base.$f(r::IdOffsetRange) = eltype(r)($f(r.parent) + r.offset)
end

# Iteration for an IdOffsetRange
@inline Base.iterate(r::IdOffsetRange, i...) = _iterate(r, i...)
# In general we iterate over the parent term by term and add the offset.
# This might have some performance degradation when coupled with bounds-checking
# See https://github.com/JuliaArrays/OffsetArrays.jl/issues/214
@inline function _iterate(r::IdOffsetRange, i...)
    ret = iterate(r.parent, i...)
    ret === nothing && return nothing
    return (eltype(r)(ret[1] + r.offset), ret[2])
end
# Base.OneTo(n) is known to be exactly equivalent to the range 1:n,
# and has no specialized iteration defined for it,
# so we may add the offset to the range directly and iterate over the result
# This gets around the performance issue described in issue #214
# We use the helper function _addoffset to evaluate the range instead of broadcasting
# just in case this makes it easy for the compiler.
@inline _iterate(r::IdOffsetRange{<:Integer, <:Base.OneTo}, i...) = iterate(_addoffset(r.parent, r.offset), i...)

@inline function Base.getindex(r::IdOffsetRange, i::Integer)
    i isa Bool && throw(ArgumentError("invalid index: $i of type Bool"))
    @boundscheck checkbounds(r, i)
    @inbounds eltype(r)(r.parent[i - r.offset] + r.offset)
end

# Logical indexing following https://github.com/JuliaLang/julia/pull/31829
#= Helper function to perform logical indxeing for boolean ranges
The code implemented is a branch-free version of the following:

    range(first(s) ? first(r) : last(r), length=Int(last(s)))

See https://github.com/JuliaArrays/OffsetArrays.jl/pull/224#discussion_r595635143

Logical indexing does not preserve indices, unlike other forms of vector indexing
=#
@inline function _getindex(r, s::AbstractUnitRange{Bool})
    range(first(r) * first(s) + last(r) * !first(s), length=Int(last(s)))
end
@inline function _getindex(r, s::StepRange{Bool})
    range(first(r) * first(s) + last(r) * !first(s), step = oneunit(step(s)), length=Int(last(s)))
end
@inline function _getindex(r, s::AbstractUnitRange)
    @inbounds rs = r.parent[_subtractoffset(s, r.offset)] .+ r.offset
    _indexedby(rs, axes(s))
end
@inline function _getindex(r, s::StepRange)
    rs = @inbounds r.parent[s .- r.offset] .+ r.offset
    _indexedby(rs, axes(s))
end

for T in [:AbstractUnitRange, :StepRange]
    @eval @inline function Base.getindex(r::IdOffsetRange, s::$T{<:Integer})
        @boundscheck checkbounds(r, s)
        return _getindex(r, s)
    end
end

# These methods are necessary to avoid ambiguity
for R in [:IIUR, :IdOffsetRange]
    @eval @inline function Base.getindex(r::IdOffsetRange, s::$R)
        @boundscheck checkbounds(r, s)
        return _getindex(r, s)
    end
end

# offset-preserve broadcasting
Broadcast.broadcasted(::Base.Broadcast.DefaultArrayStyle{1}, ::typeof(-), r::IdOffsetRange, x::Integer) =
    IdOffsetRange(r.parent .- x, r.offset)
Broadcast.broadcasted(::Base.Broadcast.DefaultArrayStyle{1}, ::typeof(+), r::IdOffsetRange, x::Integer) =
    IdOffsetRange(r.parent .+ x, r.offset)
Broadcast.broadcasted(::Base.Broadcast.DefaultArrayStyle{1}, ::typeof(+), x::Integer, r::IdOffsetRange) =
    IdOffsetRange(x .+ r.parent, r.offset)
Broadcast.broadcasted(::Base.Broadcast.DefaultArrayStyle{1}, ::typeof(big), r::IdOffsetRange) =
    IdOffsetRange(big.(r.parent), r.offset)

Base.show(io::IO, r::IdOffsetRange) = print(io, IdOffsetRange, "(values=",first(r), ':', last(r),", indices=",first(eachindex(r)),':',last(eachindex(r)), ")")

# Optimizations
@inline Base.checkindex(::Type{Bool}, inds::IdOffsetRange, i::Real) = Base.checkindex(Bool, inds.parent, i - inds.offset)

if VERSION < v"1.5.2"
    # issue 100, 133: IdOffsetRange as another index-preserving case shouldn't comtribute offsets
    # fixed by https://github.com/JuliaLang/julia/pull/37204
    @inline Base.compute_offset1(parent, stride1::Integer, dims::Tuple{Int}, inds::Tuple{IdOffsetRange}, I::Tuple) =
        Base.compute_linindex(parent, I) - stride1*first(Base.axes1(inds[1]))
end

# This was deemed "too private" to extend: see issue #184
# # Fixes an inference failure in Base.mapfirst!
# # Test: A = OffsetArray(rand(4,4), (-3,5)); R = similar(A, (1:1, 6:9)); maximum!(R, A)
# if isdefined(Base, :_firstslice)
#     Base._firstslice(i::IdOffsetRange) = IdOffsetRange(Base._firstslice(i.parent), i.offset)
# end
