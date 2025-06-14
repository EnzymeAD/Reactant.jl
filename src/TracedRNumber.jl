module TracedRNumberOverrides

using ..Reactant:
    Reactant, TracedRNumber, TracedRArray, TracedUtils, Ops, MLIR, unwrapped_eltype
using ReactantCore
using Adapt

import Base.TwicePrecision

ReactantCore.is_traced(::TracedRNumber, seen) = true
ReactantCore.is_traced(::TracedRNumber) = true

Base.getindex(a::TracedRNumber{T}) where {T} = a

Base.to_index(x::TracedRNumber{<:Integer}) = x

Base.zero(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

Base.copy(x::TracedRNumber{T}) where {T} = TracedRNumber{T}((), x.mlir_data)

function Base.eps(::Type{TracedRNumber{T}}) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, eps(T))
end

function Base.typemin(::Type{TracedRNumber{T}}) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, typemin(T))
end
function Base.typemax(::Type{TracedRNumber{T}}) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, typemax(T))
end

function Base.rtoldefault(T::Type{<:TracedRNumber})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end

function Base.isfinite(x::TracedRNumber{<:Complex})
    return isfinite(real(x)) & isfinite(imag(x))
end
Base.isfinite(x::TracedRNumber{<:AbstractFloat}) = Ops.is_finite(x)

function Base.isnan(x::TracedRNumber{<:Complex})
    return isnan(real(x)) | isnan(imag(x))
end
function Base.isnan(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return !isfinite(x) & (x != typemax(T)) & (x != typemin(T))
end

Base.isinf(x::TracedRNumber{<:Complex}) = isinf(real(x)) | isinf(imag(x))
Base.isinf(x::TracedRNumber{<:AbstractFloat}) = Ops.is_inf(x)
Base.isinf(::TracedRNumber{<:Integer}) = false

function Base.show(io::IOty, X::TracedRNumber{T}) where {T,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRNumber{", T, "}(", X.paths, ")")
end

Base.only(A::TracedRNumber{T}) where {T} = A

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

# Bool has special promotion rules in Base
function Base.promote_rule(::Type{Bool}, ::Type{TracedRNumber{T}}) where {T}
    return TracedRNumber{T}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Bool}) where {T}
    return TracedRNumber{T}
end

function Base.promote_rule(::Type{T}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{S}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(
    T::Type{<:AbstractIrrational}, ::Type{Reactant.TracedRNumber{S}}
) where {S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(
    ::Type{Reactant.TracedRNumber{S}}, T::Type{<:AbstractIrrational}
) where {S}
    return TracedRNumber{Base.promote_type(T, S)}
end

# NOTE: This is inconsistent with the behavior of `convert` but we do it since it is a very
#       common usecase
TracedRNumber{T}(x::TracedRNumber{T}) where {T} = x
function TracedRNumber{T}(x::TracedRNumber) where {T}
    return TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end
function TracedRNumber{T}(x::Number) where {T}
    return TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end

function TracedUtils.promote_to(::Type{TracedRNumber{T}}, rhs) where {T}
    if rhs isa TracedRNumber
        rhs isa TracedRNumber{T} && return rhs
        return Ops.convert(TracedRNumber{T}, rhs)
    end
    if rhs isa TracedRArray{<:Any,0}
        return TracedUtils.promote_to(
            TracedRNumber{T},
            TracedRNumber{Reactant.unwrapped_eltype(rhs)}((), rhs.mlir_data),
        )
    end
    rhs isa Number && return TracedUtils.promote_to(TracedRNumber{T}, Ops.fill(T(rhs)))
    return TracedUtils.promote_to(TracedRNumber{T}, Ops.constant(collect(rhs)))
end

function TracedUtils.promote_to(::TracedRNumber{T}, rhs) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, rhs)
end

for (aT, bT) in (
    (TracedRNumber{<:Real}, Real),
    (Real, TracedRNumber{<:Real}),
    (TracedRNumber{<:Real}, TracedRNumber{<:Real}),
)
    @eval function Base.Complex(a::$aT, b::$bT)
        T = promote_type(unwrapped_eltype(a), unwrapped_eltype(b))
        a = TracedUtils.promote_to(TracedRNumber{T}, a)
        b = TracedUtils.promote_to(TracedRNumber{T}, b)
        return Ops.complex(a, b)
    end
end

Base.Complex(x::TracedRNumber{<:Real}) = Ops.complex(x, zero(x))
Base.Complex(x::TracedRNumber{<:Complex}) = x

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
    (:(Base.:*), :multiply),
    (:(Base.:/), :divide),
    (:(Base.:^), :power),
    (:(Base.rem), :remainder),
)
    @eval function $(jlop)(
        @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
    ) where {T}
        return Ops.$(hloop)(lhs, rhs)
    end
end

function Base.rem(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
) where {T}
    return Ops.remainder(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end
function Base.rem(
    @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
) where {T}
    return Ops.remainder(TracedUtils.promote_to(TracedRNumber{T}, lhs), rhs)
end

# Based on https://github.com/JuliaLang/julia/blob/39255d47db7657950ff1c82137ecec5a70bae622/base/float.jl#L608-L617
function Base.mod(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    r = rem(x, y)
    return ifelse(r == 0, copysign(r, y), ifelse((r > 0) ⊻ (y > 0), r + y, r))
end
function Base.mod(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
) where {T}
    return mod(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end
function Base.mod(
    @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
) where {T}
    return mod(TracedUtils.promote_to(TracedRNumber{T}, lhs), rhs)
end

function Base.div(@nospecialize(lhs::TracedRNumber{T}), rhs) where {T<:Integer}
    return Ops.divide(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), rhs, ::typeof(RoundDown)
) where {T<:Integer}
    return Ops.divide(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end

function Base.:/(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
) where {T<:Integer}
    return float(lhs) / float(rhs)
end

for (jlop, hloop, hlocomp) in (
    (:(Base.:(==)), :compare, "EQ"),
    (:(Base.:(!=)), :compare, "NE"),
    (:(Base.:(>=)), :compare, "GE"),
    (:(Base.:(>)), :compare, "GT"),
    (:(Base.:(<=)), :compare, "LE"),
    (:(Base.:(<)), :compare, "LT"),
    (:(Base.isless), :compare, "LT"),
)
    @eval begin
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return Ops.compare(lhs, rhs; comparison_direction=$(hlocomp))
        end

        function $(jlop)(@nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs)) where {T}
            return $(jlop)(lhs, TracedUtils.promote_to(lhs, rhs))
        end
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(jlop)(lhs, TracedUtils.promote_to(lhs, rhs))
        end

        function $(jlop)(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber{T})) where {T}
            return $(jlop)(TracedUtils.promote_to(rhs, lhs), rhs)
        end
        function $(jlop)(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(jlop)(TracedUtils.promote_to(rhs, lhs), rhs)
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
        ) where {T1,T2}
            commonTy = TracedRNumber{Base.promote_type(T1, T2)}
            lhs = TracedUtils.promote_to(commonTy, lhs)
            rhs = TracedUtils.promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end
end

function Base.ifelse(@nospecialize(pred::TracedRNumber{Bool}), x::Number, y::Number)
    return ifelse(
        pred,
        TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(x)}, x),
        TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(y)}, y),
    )
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T1}),
    @nospecialize(y::TracedRNumber{T2})
) where {T1,T2}
    @warn "`ifelse` with different element-types in Reactant works by promoting the \
            element-type to the common type. This is semantically different from the \
            behavior of `ifelse` in Base. Use with caution" maxlog = 1
    T = promote_type(T1, T2)
    return ifelse(
        pred,
        TracedUtils.promote_to(TracedRNumber{T}, x),
        TracedUtils.promote_to(TracedRNumber{T}, y),
    )
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T})
) where {T}
    return Ops.select(pred, x, y)
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::Tuple),
    @nospecialize(y::Tuple)
)
    @assert length(x) == length(y)
    ntuple(Val(length(x))) do i
        Base.ifelse(pred, x[i], y[i])
    end
end

function Base.:*(
    x::Base.TwicePrecision{T}, y::Base.TwicePrecision{T}
) where {T<:TracedRNumber}
    zh, zl = Base.mul12(x.hi, y.hi)
    hi, lo = Base.canonicalize2(zh, (x.hi * y.lo + x.lo * y.hi) + zl)
    hi = ifelse(iszero(zh) | !isfinite(zh), zh, hi)
    lo = ifelse(iszero(zl) | !isfinite(zl), zl, lo)

    return Base.TwicePrecision{T}(hi, lo)
end

function Base.:+(
    x::Base.TwicePrecision{T}, y::Base.TwicePrecision{T}
) where {T<:TracedRNumber}
    r = x.hi + y.hi
    @trace s = if abs(x.hi) > abs(y.hi)
        begin
            (((x.hi - r) + y.hi) + y.lo) + x.lo
        end
    else
        begin
            (((y.hi - r) + x.hi) + x.lo) + y.lo
        end
    end
    return Base.TwicePrecision(Base.canonicalize2(r, s)...)
end

function Base.:*(x::TwicePrecision, v::TracedRNumber)
    @trace result = if v == 0
        TwicePrecision(x.hi * v, x.lo * v)
    else
        x * TwicePrecision(oftype(x.hi * v, v))
    end
    return result
end

for (T1, T2) in zip((Bool, Integer), (Bool, Integer))
    T = promote_type(T1, T2)
    @eval begin
        function Base.:&(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.and(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:&(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return Ops.and(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:&(x::$(T1), y::TracedRNumber{<:$(T2)})
            return Ops.and(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.or(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return Ops.or(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::$(T1), y::TracedRNumber{<:$(T2)})
            return Ops.or(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.xor(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return Ops.xor(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::$(T1), y::TracedRNumber{<:$(T2)})
            return Ops.xor(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        Base.:!(x::TracedRNumber{<:$(T1)}) = Ops.not(x)
    end
end

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRNumber{T}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
    (:(Base.abs), :abs),
    (:(Base.:-), :negate),
    (:(Base.sin), :sine),
    (:(Base.cos), :cosine),
    (:(Base.tan), :tan),
    (:(Base.tanh), :tanh),
    (:(Base.FastMath.tanh_fast), :tanh),
    (:(Base.exp), :exponential),
    (:(Base.FastMath.exp_fast), :exponential),
    (:(Base.expm1), :exponential_minus_one),
    (:(Base.log), :log),
    (:(Base.log1p), :log_plus_one),
    (:(Base.sqrt), :sqrt),
    (:(Base.acos), :acos),
    (:(Base.acosh), :acosh),
    (:(Base.asin), :asin),
    (:(Base.asinh), :asinh),
    (:(Base.atan), :atan),
    (:(Base.atanh), :atanh),
    (:(Base.sign), :sign),
    (:(Base.conj), :conj),
    (:(Base.real), :real),
    (:(Base.imag), :imag),
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber)) = Ops.$(hloop)(lhs)
end

for (jlop, hloop) in
    ((:(Base.sinpi), :sine), (:(Base.cospi), :cosine), (:(Base.tanpi), :tan))
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{T})) where {T} = Ops.$(hloop)(T(π) * lhs)
end

Base.sincospi(x::TracedRNumber{T}) where {T} = Ops.sine(T(π) * x), Ops.cosine(T(π) * x)

Base.isreal(::TracedRNumber) = false
Base.isreal(::TracedRNumber{<:Real}) = true

Base.iseven(x::TracedRNumber) = iseven(real(x))
function Base.iseven(x::TracedRNumber{<:Real})
    return iszero(
        rem(
            TracedUtils.promote_to(TracedRNumber{Int}, x),
            TracedUtils.promote_to(TracedRNumber{Int}, 2),
        ),
    )
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval function Base.clamp(x::TracedRNumber, min::$(minT), max::$(maxT))
        T = promote_type(unwrapped_eltype(x), unwrapped_eltype(min), unwrapped_eltype(max))
        return Ops.clamp(
            TracedUtils.promote_to(TracedRNumber{T}, min),
            TracedUtils.promote_to(TracedRNumber{T}, x),
            TracedUtils.promote_to(TracedRNumber{T}, max),
        )
    end
end

function Base.fill(x::TracedRNumber, dims::NTuple{N,Integer}) where {N}
    return TracedUtils.broadcast_to_size(x, dims)
end
function Base.fill(x::TracedRNumber, ::Tuple{})
    return TracedUtils.broadcast_to_size(x, ())
end

function Base.float(x::TracedRNumber{T}) where {T}
    return TracedUtils.promote_to(TracedRNumber{float(T)}, x)
end

using Reactant: ReactantFloat, ReactantInt

Base.round(A::TracedRNumber{<:ReactantFloat}) = Ops.round_nearest_even(A)
Base.round(A::TracedRNumber{<:ReactantInt}) = A
Base.floor(A::TracedRNumber{<:ReactantFloat}) = Ops.floor(A)
Base.floor(A::TracedRNumber{<:ReactantInt}) = A
Base.ceil(A::TracedRNumber{<:ReactantFloat}) = Ops.ceil(A)
Base.ceil(A::TracedRNumber{<:ReactantInt}) = A

function Base.unsafe_trunc(
    T::Type{<:Reactant.ReactantInt}, x::TracedRNumber{<:Reactant.ReactantFloat}
)
    return Ops.convert(TracedRNumber{T}, x)
end

for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for Tf in (Float16, Float32, Float64)
        if Ti <: Unsigned || sizeof(Ti) < sizeof(Tf)
            # Here `Tf(typemin(Ti))-1` is exact, so we can compare the lower-bound
            # directly. `Tf(typemax(Ti))+1` is either always exactly representable, or
            # rounded to `Inf` (e.g. when `Ti==UInt128 && Tf==Float32`).
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRNumber{$Tf})
                    # TODO throw error within traced
                    # if $(Tf(typemin(Ti))-one(Tf)) < x < $(Tf(typemax(Ti))+one(Tf))
                    return Base.unsafe_trunc($Ti, x)
                    # else
                    #     throw(Base.InexactError(:trunc, $Ti, x))
                    # end
                end
            end
        else
            # Here `eps(Tf(typemin(Ti))) > 1`, so the only value which can be truncated to
            # `Tf(typemin(Ti)` is itself. Similarly, `Tf(typemax(Ti))` is inexact and will
            # be rounded up. This assumes that `Tf(typemin(Ti)) > -Inf`, which is true for
            # these types, but not for `Float16` or larger integer types.
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRNumber{$Tf})
                    # TODO throw error within traced
                    # if $(Tf(typemin(Ti))) <= x < $(Tf(typemax(Ti)))
                    return Base.unsafe_trunc($Ti, x)
                    # else
                    #     throw(Base.InexactError(:trunc, $Ti, x))
                    # end
                end
            end
        end
    end
end

# matches convert methods
# also determines floor, ceil, round
Base.trunc(::Type{Signed}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(Int, x)
Base.trunc(::Type{Unsigned}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(UInt, x)
Base.trunc(::Type{Integer}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(Int, x)

function Base.getindex(
    r::Union{Base.StepRangeLen,Base.LinRange}, i::TracedRNumber{<:Integer}
)
    @inline
    i isa TracedRNumber{Bool} && throw(ArgumentError("invalid index: $i of type Bool"))
    # @boundscheck checkbounds(r, i)
    return Base.unsafe_getindex(r, i)
end

function unitrange_last(start::Integer, stop::Integer)
    return ifelse(stop >= start, stop, convert(typeof(stop), start - oneunit(start - stop)))
end
function unitrange_last(start, stop)
    return ifelse(
        stop >= start,
        convert(typeof(stop), start + floor(stop - start)),
        convert(typeof(stop), start - oneunit(start - stop)),
    )
end

struct TracedUnitRange{T} <: AbstractUnitRange{T}
    start::T
    stop::T
    function TracedUnitRange{T}(start::T, stop::T) where {T}
        return new(start, unitrange_last(start, stop))
    end
end
function Adapt.parent_type(::Type{TracedUnitRange{T}}) where {T}
    return TracedUnitRange{T}
end
function TracedUnitRange{T}(start, stop) where {T}
    return TracedUnitRange{T}(convert(T, start), convert(T, stop))
end
TracedUnitRange(start::T, stop::T) where {T} = TracedUnitRange{T}(start, stop)
function TracedUnitRange(start, stop)
    startstop_promoted = promote(start, stop)
    not_sametype((start, stop), startstop_promoted)
    return TracedUnitRange(startstop_promoted...)
end
function Base._in_unit_range(
    v::TracedUnitRange, val, i::Union{Integer,TracedRNumber{<:Integer}}
)
    return (i > 0) & (val <= v.stop) & (val >= v.start)
end

function _traced_unitrange_getindex(v::TracedUnitRange{T}, i) where {T}
    val = convert(T, v.start + (i - oneunit(i)))
    # TODO: we should have error messages at some point.
    # @boundscheck Base._in_unit_range(v, val, i) || throw_boundserror(v, i)
    return val
end

function Base._getindex(v::TracedUnitRange, i::TracedRNumber{<:Integer})
    return _traced_unitrange_getindex(v, i)
end
Base.getindex(v::TracedUnitRange, i::Integer) = _traced_unitrange_getindex(v, i)
Base.getindex(r::TracedUnitRange, i::TracedRNumber) = Base._getindex(r, i)
function Base.getindex(r::Base.UnitRange, i::I) where {I<:TracedRNumber{<:Integer}}
    val = convert(I, r.start + (i - oneunit(i)))
    # TODO: we should have error messages at some point.
    # @boundscheck Base._in_unit_range(v, val, i) || throw_boundserror(v, i)
    return val
end

function Base.promote_rule(
    a::Type{TracedUnitRange{T1}}, b::Type{TracedUnitRange{T2}}
) where {T1,T2}
    return el_same(promote_type(T1, T2), a, b)
end
TracedUnitRange{T}(r::TracedUnitRange{T}) where {T<:Real} = r
TracedUnitRange{T}(r::TracedUnitRange) where {T<:Real} = TracedUnitRange{T}(r.start, r.stop)

function Base.promote_rule(
    a::Type{TracedUnitRange{T1}}, ::Type{UR}
) where {T1,UR<:AbstractUnitRange}
    return promote_rule(a, TracedUnitRange{eltype(UR)})
end
function TracedUnitRange{T}(r::AbstractUnitRange) where {T<:Real}
    return TracedUnitRange{T}(first(r), last(r))
end
TracedUnitRange(r::AbstractUnitRange) = TracedUnitRange(first(r), last(r))

@inline function Base.length(r::TracedUnitRange{TracedRNumber{T}}) where {T}
    start, stop = first(r), last(r)
    a = Base.oneunit(Base.zero(stop) - Base.zero(start))
    if a isa Signed
        # Signed are allowed to go negative
        Ops.select(stop >= start, a + stop - start, a)
    else
        Ops.select(stop >= start, a + stop - start, zero(a))
    end
end

function Base._reshape(v::TracedUnitRange, dims::Dims{1})
    Base.require_one_based_indexing(v)
    len = dims[1]
    # TODO support errors
    # len == length(v) || Base._throw_dmrs(length(v), "length", len)
    return v
end
function Base._reshape(parent::TracedUnitRange, dims::Dims)
    n = length(parent)
    # TODO support errors
    # prod(dims) == n || Base._throw_dmrs(n, "size", dims)
    return Base.__reshape((parent, IndexStyle(parent)), dims)
end

AbstractUnitRange{T}(r::TracedUnitRange) where {T} = TracedUnitRange{T}(r)

struct TracedStepRangeLen{T,R,S,L} <: AbstractRange{T}
    ref::R
    step::S
    len::L
    offset::L
end

function Base.Array(x::TracedStepRangeLen{<:Reactant.AbstractConcreteNumber})
    return StepRangeLen(
        Reactant.to_number(x.ref),
        Reactant.to_number(x.step),
        Reactant.to_number(x.len),
        Reactant.to_number(x.offset),
    )
end

function Adapt.parent_type(::Type{TracedStepRangeLen{T,R,S,L}}) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}
end

# constructors and interface implementation copied from range.jl
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

function _tracedsteprangelen_unsafe_getindex(
    r::AbstractRange{T}, i::Union{I,TracedRNumber{I}}
) where {T,I}
    finalT = T
    offsetT = typeof(r.offset)
    if i isa TracedRNumber
        if !(T <: TracedRNumber)
            finalT = TracedRNumber{T}
        end
        if !(r.offset isa TracedRNumber)
            offsetT = TracedRNumber{offsetT}
        end
    end
    u = convert(offsetT, i) - r.offset
    return finalT(r.ref + u * r.step)
end
function Base.unsafe_getindex(r::TracedStepRangeLen, i::Integer)
    return _tracedsteprangelen_unsafe_getindex(r, i)
end
function Base.unsafe_getindex(r::TracedStepRangeLen, i::TracedRNumber{<:Integer})
    return _tracedsteprangelen_unsafe_getindex(r, i)
end
Base.getindex(r::TracedStepRangeLen, i::TracedRNumber) = Base.unsafe_getindex(r, i)
function getindex(r::TracedStepRangeLen{T}, s::OrdinalRange{S}) where {T,S<:Integer}
    @inline
    @boundscheck checkbounds(r, s)

    len = length(s)
    sstep = Base.step_hp(s)
    rstep = Base.step_hp(r)
    L = typeof(len)
    if S === Bool
        rstep *= one(sstep)
        if len == 0
            return TracedStepRangeLen{T}(first(r), rstep, zero(L), oneunit(L))
        elseif len == 1
            if first(s)
                return TracedStepRangeLen{T}(first(r), rstep, oneunit(L), oneunit(L))
            else
                return TracedStepRangeLen{T}(first(r), rstep, zero(L), oneunit(L))
            end
        else # len == 2
            return TracedStepRangeLen{T}(last(r), rstep, oneunit(L), oneunit(L))
        end
    else
        # Find closest approach to offset by s
        ind = LinearIndices(s)
        offset = L(
            max(min(1 + round(L, (r.offset - first(s)) / sstep), last(ind)), first(ind))
        )
        ref = Base._getindex_hiprec(r, first(s) + (offset - oneunit(offset)) * sstep)
        return TracedStepRangeLen{T}(ref, rstep * sstep, len, offset)
    end
end
function Base._getindex_hiprec(r::TracedStepRangeLen, i::Integer)  # without rounding by T
    u = oftype(r.offset, i) - r.offset
    return r.ref + u * r.step
end
function Base.:(==)(r::T, s::T) where {T<:TracedStepRangeLen}
    return (isempty(r) & isempty(s)) |
           ((first(r) == first(s)) & (length(r) == length(s)) & (last(r) == last(s)))
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
TracedStepRangeLen{T,R,S,L}(r::TracedStepRangeLen{T,R,S,L}) where {T,R,S,L} = r
function TracedStepRangeLen{T,R,S,L}(r::TracedStepRangeLen) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}(
        convert(R, r.ref), convert(S, r.step), convert(L, r.len), convert(L, r.offset)
    )
end
function TracedStepRangeLen{T}(r::TracedStepRangeLen) where {T}
    return TracedStepRangeLen(convert(T, r.ref), convert(T, r.step), r.len, r.offset)
end
function Base.promote_rule(
    a::Type{TracedStepRangeLen{T,R,S,L}}, ::Type{OR}
) where {T,R,S,L,OR<:AbstractRange}
    return promote_rule(a, TracedStepRangeLen{eltype(OR),eltype(OR),eltype(OR),Int})
end
function TracedStepRangeLen{T,R,S,L}(r::AbstractRange) where {T,R,S,L}
    return TracedStepRangeLen{T,R,S,L}(R(first(r)), S(step(r)), length(r))
end
function TracedStepRangeLen{T}(r::AbstractRange) where {T}
    return TracedStepRangeLen(T(first(r)), T(step(r)), length(r))
end
TracedStepRangeLen(r::AbstractRange) = TracedStepRangeLen{eltype(r)}(r)

function Base.promote_rule(
    ::Type{LinRange{A,L}}, b::Type{TracedStepRangeLen{T2,R2,S2,L2}}
) where {A,L,T2,R2,S2,L2}
    return promote_rule(TracedStepRangeLen{A,A,A,L}, b)
end

function Base._reverse(r::TracedStepRangeLen, ::Colon)
    # If `r` is empty, `length(r) - r.offset + 1 will be nonpositive hence
    # invalid. As `reverse(r)` is also empty, any offset would work so we keep
    # `r.offset`
    offset = isempty(r) ? r.offset : length(r) - r.offset + 1
    return typeof(r)(r.ref, negate(r.step), length(r), offset)
end

# TODO: +, - for TracedStepRangeLen (see Base._define_range_op)

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.TracedRNumber}
    return (T(x.hi) + T(x.lo))::T
end

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.ConcreteRNumber}
    return Reactant.ConcreteRNumber(T(x.hi) - T(x.lo))::T
end

Base.nbitslen(r::TracedStepRangeLen) = Base.nbitslen(eltype(r), length(r), r.offset)
function TracedStepRangeLen(
    ref::TwicePrecision{T}, step::TwicePrecision{T}, len, offset=1
) where {T}
    return TracedStepRangeLen{T,TwicePrecision{T},TwicePrecision{T}}(ref, step, len, offset)
end
function Base.step(r::TracedStepRangeLen{T,TwicePrecision{T},TwicePrecision{T}}) where {T}
    return T(r.step)
end

# This assumes that r.step has already been split so that (0:len-1)*r.step.hi is exact
function Base.unsafe_getindex(
    r::Union{
        Base.StepRangeLen{T,<:Base.TwicePrecision,<:Base.TwicePrecision},
        TracedStepRangeLen{
            T,<:Base.TwicePrecision,<:Base.TwicePrecision,<:Base.TwicePrecision
        },
    },
    i::TracedRNumber{<:Integer},
) where {T}
    # Very similar to _getindex_hiprec, but optimized to avoid a 2nd call to add12
    @inline
    i isa TracedRNumber{Bool} && throw(ArgumentError("invalid index: $i of type Bool"))
    OT = if r.offset isa TracedRNumber
        typeof(r.offset)
    else
        TracedRNumber{typeof(r.offset)}
    end
    u = Base.convert(OT, i)::OT - r.offset
    shift_hi, shift_lo = u * r.step.hi, u * r.step.lo
    x_hi, x_lo = Base.add12(r.ref.hi, shift_hi)
    T2 = if T isa TracedRNumber
        T
    else
        TracedRNumber{T}
    end
    return T2(x_hi + (x_lo + (shift_lo + r.ref.lo)))
end

function Base.searchsortedfirst(
    a::AbstractRange{<:Union{Real,TracedRNumber}},
    x::TracedRNumber{<:Real},
    o::Base.DirectOrdering,
)::TracedRNumber{keytype(a)}

    # require_one_based_indexing(a)
    f, h, l = first(a), step(a), last(a)
    n = round(Int, (x - f) / h + 1)

    return ifelse(
        !Base.Order.lt(o, f, x),
        1,
        ifelse(
            (h == 0) | Base.Order.lt(o, l, x),
            length(a) + 1,
            ifelse(Base.Order.lt(o, a[n], x), n + 1, n),
        ),
    )
end

function Base.searchsortedfirst(
    a::AbstractRange{<:TracedRNumber}, x::Real, o::Base.DirectOrdering
)::TracedRNumber{keytype(a)}
    return Base.searchsortedfirst(a, TracedRNumber{typeof(x)}(x), o)
end

function Base.round(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.round(x))
end
function Base.floor(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.floor(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.ceil(x))
end

# Concatenation. Numbers in Julia are handled in a much less generic fashion than arrays
Base.vcat(x::TracedRNumber...) = Base.typed_vcat(Base.promote_eltypeof(x...), x...)
function Base.typed_vcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_vcat(T, map(Base.Fix2(TracedUtils.broadcast_to_size, (1,)), x)...)
end

Base.hcat(x::TracedRNumber...) = Base.typed_hcat(Base.promote_eltypeof(x...), x...)
function Base.typed_hcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_hcat(T, map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), x)...)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, xs::TracedRNumber...)
    return Base.typed_hvcat(Base.promote_eltypeof(xs...), rows, xs...)
end
function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Vararg{Int}}, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvcat(T, rows, xs...)
end

function Base.hvncat(dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...)
    return Base.typed_hvncat(Base.promote_eltypeof(xs...), dims, row_first, xs...)
end
function Base.typed_hvncat(
    ::Type{T}, dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvncat(T, dims, row_first, xs...)
end

for (Ti, Tf) in ((Int16, Float16), (Int32, Float32), (Int64, Float64))
    @eval begin
        Base.signbit(x::TracedRNumber{$(Ti)}) = x < 0
        Base.signbit(x::TracedRNumber{$(Tf)}) = signbit(Ops.bitcast_convert($(Ti), x))
    end
end
Base.signbit(::TracedRNumber{<:Unsigned}) = ConcretePJRTNumber(false)

Base.copysign(x::TracedRNumber, y::TracedRNumber) = ifelse(signbit(y), -1, 1) * abs(x)
function Base.copysign(x::TracedRNumber{T}, y::S) where {T,S<:Number}
    return copysign(x, TracedUtils.promote_to(TracedRNumber{S}, y))
end
function Base.copysign(x::S, y::TracedRNumber{T}) where {S<:Number,T}
    return copysign(TracedUtils.promote_to(TracedRNumber{S}, x), y)
end

end # module TracedRNumberOverrides
