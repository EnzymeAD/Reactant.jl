module TracedRangeOverrides

using ..Reactant: Reactant, AbstractConcreteNumber, TracedRNumber
import ..Reactant: TracedStepRangeLen, TracedUnitRange
using ..Ops: @opcall

using Base: TwicePrecision, IndexStyle

# TracedStepRangeLen

function Base.Array(x::TracedStepRangeLen{<:AbstractConcreteNumber})
    return StepRangeLen(
        Reactant.to_number(x.ref),
        Reactant.to_number(x.step),
        Reactant.to_number(x.len),
        Reactant.to_number(x.offset),
    )
end

function TracedStepRangeLen{T,R,S}(ref::R, step::S, len, offset=1) where {T,R,S}
    return TracedStepRangeLen{T,R,S,typeof(len)}(ref, step, len, offset)
end
function TracedStepRangeLen(ref::R, step::S, len, offset=1) where {R,S}
    return TracedStepRangeLen{typeof(ref + zero(step)),R,S,typeof(len)}(
        ref, step, len, offset
    )
end
function TracedStepRangeLen{T}(
    ref::R, step::S, len::Integer, offset::Integer=1
) where {T,R,S}
    return TracedStepRangeLen{T,R,S,typeof(len)}(ref, step, len, offset)
end

TracedStepRangeLen{T,R,S,L}(r::TracedStepRangeLen{T,R,S,L}) where {T,R,S,L} = r
function TracedStepRangeLen{T,R,S,L}(r::TracedStepRangeLen) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}(
        convert(R, r.ref), convert(S, r.step), convert(L, r.len), convert(L, r.offset)
    )
end
function TracedStepRangeLen{T}(r::TracedStepRangeLen) where {T}
    return TracedStepRangeLen(convert(T, r.ref), convert(T, r.step), r.len, r.offset)
end
function TracedStepRangeLen{T,R,S,L}(r::AbstractRange) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}(R(first(r)), S(step(r)), length(r))
end
function TracedStepRangeLen{T}(r::AbstractRange) where {T}
    return TracedStepRangeLen(T(first(r)), T(step(r)), length(r))
end
TracedStepRangeLen(r::AbstractRange) = TracedStepRangeLen{eltype(r)}(r)
function TracedStepRangeLen(
    ref::TwicePrecision{T}, step::TwicePrecision{T}, len, offset=1
) where {T}
    return TracedStepRangeLen{T,TwicePrecision{T},TwicePrecision{T}}(ref, step, len, offset)
end

Base.isempty(r::TracedStepRangeLen) = length(r) == 0
Base.step(r::TracedStepRangeLen) = r.step
Base.step_hp(r::TracedStepRangeLen) = r.step
Base.length(r::TracedStepRangeLen) = r.len
Base.first(r::TracedStepRangeLen) = Base.unsafe_getindex(r, 1)
Base.last(r::TracedStepRangeLen) = Base.unsafe_getindex(r, r.len)
function Base.iterate(r::TracedStepRangeLen, i::Integer=1)
    @inline
    i += oneunit(i)
    length(r) < i && return nothing
    return Base.unsafe_getindex(r, i), i
end

# TODO: if there ever comes a ReactantStepRange:
# ==(r::Union{StepRange{T},StepRangeLen{T,T}}, s::Union{StepRange{T},StepRangeLen{T,T}}) where {T}

function Base.:-(r::TracedStepRangeLen{T,R,S,L}) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}(-r.ref, -r.step, r.len, r.offset)
end

# TODO: promotion from StepRangeLen{T} to TracedStepRangeLen{T}?
function Base.promote_rule(
    ::Type{TracedStepRangeLen{T1,R1,S1,L1}}, ::Type{TracedStepRangeLen{T2,R2,S2,L2}}
) where {T1,T2,R1,R2,S1,S2,L1,L2}
    R, S, L = promote_type(R1, R2), promote_type(S1, S2), promote_type(L1, L2)
    return Base.el_same(
        promote_type(T1, T2), TracedStepRangeLen{T1,R,S,L}, TracedStepRangeLen{T2,R,S,L}
    )
end
function Base.promote_rule(
    a::Type{TracedStepRangeLen{T,R,S,L}}, ::Type{OR}
) where {T,R,S,L,OR<:AbstractRange}
    return promote_rule(a, TracedStepRangeLen{eltype(OR),eltype(OR),eltype(OR),Int})
end

function Base.promote_rule(
    ::Type{LinRange{A,L}}, b::Type{TracedStepRangeLen{T2,R2,S2,L2}}
) where {A,L,T2,R2,S2,L2}
    return promote_rule(TracedStepRangeLen{A,A,A,L}, b)
end

Base.nbitslen(r::TracedStepRangeLen) = Base.nbitslen(eltype(r), length(r), r.offset)

function Base.step(r::TracedStepRangeLen{T,TwicePrecision{T},TwicePrecision{T}}) where {T}
    return T(r.step)
end

function Base._reverse(r::TracedStepRangeLen, ::Colon)
    # If `r` is empty, `length(r) - r.offset + 1 will be nonpositive hence
    # invalid. As `reverse(r)` is also empty, any offset would work so we keep
    # `r.offset`
    offset = isempty(r) ? r.offset : length(r) - r.offset + 1
    return typeof(r)(r.ref, negate(r.step), length(r), offset)
end

function Base.:(==)(r::T, s::T) where {T<:TracedStepRangeLen}
    return (isempty(r) & isempty(s)) |
           ((first(r) == first(s)) & (length(r) == length(s)) & (last(r) == last(s)))
end

# TODO: +, - for TracedStepRangeLen (see Base._define_range_op)

# TracedUnitRange
AbstractUnitRange{T}(r::TracedUnitRange) where {T} = TracedUnitRange{T}(r)

function TracedUnitRange{T}(start, stop) where {T}
    return TracedUnitRange{T}(convert(T, start), convert(T, stop))
end
TracedUnitRange(start::T, stop::T) where {T} = TracedUnitRange{T}(start, stop)
function TracedUnitRange(start, stop)
    startstop_promoted = promote(start, stop)
    not_sametype((start, stop), startstop_promoted)
    return TracedUnitRange(startstop_promoted...)
end
TracedUnitRange{T}(r::TracedUnitRange{T}) where {T<:Real} = r
TracedUnitRange{T}(r::TracedUnitRange) where {T<:Real} = TracedUnitRange{T}(r.start, r.stop)
function TracedUnitRange{T}(r::AbstractUnitRange) where {T<:Real}
    return TracedUnitRange{T}(first(r), last(r))
end
TracedUnitRange(r::AbstractUnitRange) = TracedUnitRange(first(r), last(r))

function Base.promote_rule(
    a::Type{TracedUnitRange{T1}}, b::Type{TracedUnitRange{T2}}
) where {T1,T2}
    return el_same(promote_type(T1, T2), a, b)
end

function Base.promote_rule(
    a::Type{TracedUnitRange{T1}}, ::Type{UR}
) where {T1,UR<:AbstractUnitRange}
    return promote_rule(a, TracedUnitRange{eltype(UR)})
end

function Base._in_unit_range(
    v::TracedUnitRange, val, i::Union{Integer,TracedRNumber{<:Integer}}
)
    return (i > 0) & (val <= v.stop) & (val >= v.start)
end

@inline function Base.length(r::TracedUnitRange{TracedRNumber{T}}) where {T}
    start, stop = first(r), last(r)
    a = Base.oneunit(Base.zero(stop) - Base.zero(start))
    if a isa Signed
        # Signed are allowed to go negative
        @opcall select(stop >= start, a + stop - start, a)
    else
        @opcall select(stop >= start, a + stop - start, zero(a))
    end
end

function Base._reshape(v::TracedUnitRange, dims::Dims{1})
    Base.require_one_based_indexing(v)
    # len = dims[1]
    # TODO support errors
    # len == length(v) || Base._throw_dmrs(length(v), "length", len)
    return v
end
function Base._reshape(parent::TracedUnitRange, dims::Dims)
    # n = length(parent)
    # TODO support errors
    # prod(dims) == n || Base._throw_dmrs(n, "size", dims)
    return Base.__reshape((parent, IndexStyle(parent)), dims)
end

function (C::Base.Colon)(start::TracedRNumber{T}, stop::TracedRNumber{T}) where {T}
    return TracedUnitRange(start, stop)
end
function (C::Base.Colon)(start::TracedRNumber{T}, stop::T) where {T}
    return C(start, TracedRNumber{T}(stop))
end
function (C::Base.Colon)(start::T, stop::TracedRNumber{T}) where {T}
    return C(TracedRNumber{T}(start), stop)
end
end
