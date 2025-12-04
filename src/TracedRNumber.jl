module TracedRNumberOverrides

using ..Reactant: Reactant, TracedRNumber, TracedRArray, Ops, unwrapped_eltype
using ..Ops: @opcall
using ReactantCore: ReactantCore, @trace
using Adapt: Adapt

# This isn't technically necessary in this module, but this type used to be
# defined in this module so we keep this alias here for compatibility.  TODO:
# can be removed in future breaking version of Reactant.
const TracedStepRangeLen = Reactant.TracedStepRangeLen

import Base.TwicePrecision

ReactantCore.is_traced(::TracedRNumber, seen) = true
ReactantCore.is_traced(::TracedRNumber) = true

Base.to_index(x::TracedRNumber{<:Integer}) = x

Base.zero(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

Base.copy(x::TracedRNumber{T}) where {T} = TracedRNumber{T}((), x.mlir_data)

function Base.eps(::Type{TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, eps(T))
end
Base.eps(x::TracedRNumber{T}) where {T} = eps(typeof(x))

function Base.typemin(::Type{TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemin(T))
end
Base.typemin(x::TracedRNumber{T}) where {T} = typemin(typeof(x))

function Base.typemax(::Type{TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemax(T))
end
Base.typemax(x::TracedRNumber{T}) where {T} = typemax(typeof(x))

function Base.nextfloat(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return @opcall next_after(x, typemax(x))
end

function Base.prevfloat(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return @opcall next_after(x, typemin(x))
end

function Base.rtoldefault(T::Type{<:TracedRNumber})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end

function Base.isfinite(x::TracedRNumber{<:Complex})
    return isfinite(real(x)) & isfinite(imag(x))
end
Base.isfinite(x::TracedRNumber{<:AbstractFloat}) = @opcall is_finite(x)

function Base.isnan(x::TracedRNumber{<:Complex})
    return isnan(real(x)) | isnan(imag(x))
end
function Base.isnan(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return !isfinite(x) & (x != typemax(T)) & (x != typemin(T))
end

Base.isinf(x::TracedRNumber{<:Complex}) = isinf(real(x)) | isinf(imag(x))
Base.isinf(x::TracedRNumber{<:AbstractFloat}) = @opcall is_inf(x)
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

function Base.promote_rule(::Type{Nothing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Nothing}) where {T}
    return Union{Nothing,TracedRNumber{T}}
end

function Base.promote_rule(::Type{Missing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Missing}) where {T}
    return Union{Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    ::Type{Union{Nothing,Missing}}, ::Type{TracedRNumber{S}}
) where {S}
    return Union{Nothing,Missing,TracedRNumber{S}}
end

function Base.promote_rule(
    ::Type{TracedRNumber{T}}, ::Type{Union{Nothing,Missing}}
) where {T}
    return Union{Nothing,Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    T::Type{>:Union{Nothing,Missing}}, ::Type{TracedRNumber{S}}
) where {S}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(
    ::Type{TracedRNumber{T}}, S::Type{>:Union{Nothing,Missing}}
) where {T}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(T::Type{>:Missing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, S::Type{>:Missing}) where {T}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(T, S))}}
end

function Base.promote_rule(::Type{>:Nothing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, S::Type{>:Nothing}) where {T}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(T, S))}}
end

function Base.promote_rule(::Type{TwicePrecision{T}}, ::Type{TracedRNumber{S}}) where {T,S}
    return TwicePrecision{Base.promote_type(T, TracedRNumber{S})}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{TwicePrecision{S}}) where {T,S}
    return TwicePrecision{Base.promote_type(TracedRNumber{T}, S)}
end

# NOTE: This is inconsistent with the behavior of `convert` but we do it since it is a very
#       common usecase
TracedRNumber{T}(x::TracedRNumber{T}) where {T} = x
function TracedRNumber{T}(x::TracedRNumber) where {T}
    return Reactant.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end
function TracedRNumber{T}(x::Number) where {T}
    return Reactant.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end

for T in Base.uniontypes(Reactant.ReactantFloat8)
    @eval TracedRNumber{T}(x::$T) where {T} = Reactant.promote_to(TracedRNumber{T}, x)
end

for (aT, bT) in (
    (TracedRNumber{<:Real}, Real),
    (Real, TracedRNumber{<:Real}),
    (TracedRNumber{<:Real}, TracedRNumber{<:Real}),
)
    @eval function Base.Complex(a::$aT, b::$bT)
        T = promote_type(unwrapped_eltype(a), unwrapped_eltype(b))
        a = Reactant.promote_to(TracedRNumber{T}, a)
        b = Reactant.promote_to(TracedRNumber{T}, b)
        return @opcall complex(a, b)
    end
end

Base.Complex(x::TracedRNumber{<:Real}) = @opcall complex(x, zero(x))
Base.Complex(x::TracedRNumber{<:Complex}) = x

# Base.complex
Base.complex(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{complex(T)}
Base.complex(x::TracedRNumber{<:Real}) = complex(x, zero(x))
function Base.complex(x::TracedRNumber{<:Real}, y::TracedRNumber{<:Real})
    T = promote_type(unwrapped_eltype(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRNumber{<:Real}, y::Real)
    T = promote_type(unwrapped_eltype(x), typeof(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::Real, y::TracedRNumber{<:Real})
    T = promote_type(typeof(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return @opcall complex(x, y)
end
Base.complex(x::TracedRNumber{T}) where {T<:Complex} = x

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
    @eval function $(jlop)(lhs::TracedRNumber{T}, rhs::TracedRNumber{T}) where {T}
        return @opcall $(hloop)(lhs, rhs)
    end
end

function Base.:*(x::TracedRNumber{T}, z::Complex{Bool}) where {T<:Real}
    # this is to support multiplication by im (Complex{Bool}(false, true))
    z_re, z_im = real(z), imag(z)
    res_re = z_re ? x : zero(x)
    res_im = z_im ? x : zero(x)
    return Complex(res_re, res_im)
end
Base.:*(z::Complex{Bool}, x::TracedRNumber{T}) where {T<:Real} = x * z

# Based on https://github.com/JuliaLang/julia/blob/39255d47db7657950ff1c82137ecec5a70bae622/base/float.jl#L608-L617
function Base.mod(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    r = rem(x, y)
    return ifelse(r == 0, copysign(r, y), ifelse((r > 0) ⊻ (y > 0), r + y, r))
end

function Base.mod1(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    m = mod(x, y)
    return ifelse(m == 0, y, m)
end

for op in (:mod, :mod1, :rem)
    @eval begin
        function Base.$op(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(op)(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
        end
        function Base.$op(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(op)(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
        end
    end
end

function Base.div(@nospecialize(lhs::TracedRNumber{T}), rhs) where {T<:Integer}
    return @opcall divide(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), rhs, ::typeof(RoundDown)
) where {T<:Integer}
    return @opcall divide(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), ::Missing, ::typeof(RoundDown)
) where {T<:Integer}
    return missing
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
            return @opcall compare(lhs, rhs; comparison_direction=$(hlocomp))
        end

        # ambiguity fixes
        $(jlop)(@nospecialize(lhs::TracedRNumber), @nospecialize(::Missing)) = missing
        $(jlop)(@nospecialize(::Missing), @nospecialize(rhs::TracedRNumber)) = missing

        function $(jlop)(@nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs)) where {T}
            return $(jlop)(lhs, Reactant.promote_to(lhs, rhs))
        end
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(jlop)(lhs, Reactant.promote_to(lhs, rhs))
        end

        function $(jlop)(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber{T})) where {T}
            return $(jlop)(Reactant.promote_to(rhs, lhs), rhs)
        end
        function $(jlop)(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(jlop)(Reactant.promote_to(rhs, lhs), rhs)
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
        ) where {T1,T2}
            commonTy = TracedRNumber{Base.promote_type(T1, T2)}
            lhs = Reactant.promote_to(commonTy, lhs)
            rhs = Reactant.promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end
end

function Base.ifelse(@nospecialize(pred::TracedRNumber{Bool}), x::Number, y::Number)
    return ifelse(
        pred,
        Reactant.promote_to(TracedRNumber{unwrapped_eltype(x)}, x),
        Reactant.promote_to(TracedRNumber{unwrapped_eltype(y)}, y),
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
        Reactant.promote_to(TracedRNumber{T}, x),
        Reactant.promote_to(TracedRNumber{T}, y),
    )
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T})
) where {T}
    return @opcall select(pred, x, y)
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
            return @opcall and(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:&(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return @opcall and(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:&(x::$(T1), y::TracedRNumber{<:$(T2)})
            return @opcall and(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return @opcall or(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return @opcall or(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::$(T1), y::TracedRNumber{<:$(T2)})
            return @opcall or(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::TracedRNumber{<:$(T1)}, y::$(T2))
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.xor(x::$(T1), y::TracedRNumber{<:$(T2)})
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{$(T)}, x),
                Reactant.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        Base.:!(x::TracedRNumber{<:$(T1)}) = @opcall not(x)
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
    (:(Base.cbrt), :cbrt),
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
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber)) = @opcall $(hloop)(lhs)
end

# Degree-based trigonometric wrappers for TracedRNumber
# These convert to radians internally so Reactant can lower to
# StableHLO-supported radian trigonometric operations.

Base.sind(x::TracedRNumber) = sin(deg2rad(x))
Base.cosd(x::TracedRNumber) = cos(deg2rad(x))
Base.tand(x::TracedRNumber) = tan(deg2rad(x))
Base.cscd(x::TracedRNumber) = 1 / sind(x)
Base.secd(x::TracedRNumber) = 1 / cosd(x)
Base.cotd(x::TracedRNumber) = 1 / tand(x)

Base.asind(x::TracedRNumber) = rad2deg(asin(x))
Base.acosd(x::TracedRNumber) = rad2deg(acos(x))
Base.atand(x::TracedRNumber) = rad2deg(atan(x))

Base.atan(y::TracedRNumber, x::TracedRNumber) = @opcall atan2(y, x)
Base.atand(y::TracedRNumber, x::TracedRNumber) = rad2deg(atan(y, x))

Base.acscd(x::TracedRNumber) = rad2deg(asin(1 / x))
Base.asecd(x::TracedRNumber) = rad2deg(acos(1 / x))
Base.acotd(x::TracedRNumber) = rad2deg(atan(1 / x))

for (jlop, hloop) in (
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
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{<:Integer})) =
        @opcall $(hloop)(float(lhs))
end

for (jlop, hloop) in
    ((:(Base.sinpi), :sine), (:(Base.cospi), :cosine), (:(Base.tanpi), :tan))
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{T})) where {T} =
        @opcall $(hloop)(T(π) * lhs)
end

function Base.sincospi(x::TracedRNumber{T}) where {T}
    return @opcall(sine(T(π) * x)), @opcall(cosine(T(π) * x))
end

@noinline Base.Math.log10(x::TracedRNumber) = Base.Math._log(x, Val(10), :log10)
@noinline Base.Math.log2(x::TracedRNumber) = Base.Math._log(x, Val(2), :log2)
Base.Math._log(x::TracedRNumber, base, ::Symbol) = log(x) / log(Reactant._unwrap_val(base))

Base.isreal(::TracedRNumber) = false
Base.isreal(::TracedRNumber{<:Real}) = true

Base.isinteger(x::TracedRNumber{<:Integer}) = true
Base.isinteger(x::TracedRNumber{<:AbstractFloat}) = x - trunc(x) == zero(x)

Base.isodd(x::TracedRNumber) = isodd(real(x))
function Base.isodd(x::TracedRNumber{<:Real})
    return (
        isinteger(x) &
        !iszero(
            rem(
                Reactant.promote_to(TracedRNumber{Int}, x),
                Reactant.promote_to(TracedRNumber{Int}, 2),
            ),
        )
    )
end

Base.iseven(x::TracedRNumber) = iseven(real(x))
function Base.iseven(x::TracedRNumber{<:Real})
    return (
        isinteger(x) & iszero(
            rem(
                Reactant.promote_to(TracedRNumber{Int}, x),
                Reactant.promote_to(TracedRNumber{Int}, 2),
            ),
        )
    )
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval function Base.clamp(x::TracedRNumber, min::$(minT), max::$(maxT))
        T = promote_type(unwrapped_eltype(x), unwrapped_eltype(min), unwrapped_eltype(max))
        return @opcall clamp(
            Reactant.promote_to(TracedRNumber{T}, min),
            Reactant.promote_to(TracedRNumber{T}, x),
            Reactant.promote_to(TracedRNumber{T}, max),
        )
    end
end

function Base.float(x::TracedRNumber{T}) where {T}
    return Reactant.promote_to(TracedRNumber{float(T)}, x)
end

using Reactant: ReactantFloat, ReactantInt

Base.round(A::TracedRNumber{<:ReactantFloat}) = @opcall round_nearest_even(A)
Base.round(A::TracedRNumber{<:ReactantInt}) = A
Base.floor(A::TracedRNumber{<:ReactantFloat}) = @opcall floor(A)
Base.floor(A::TracedRNumber{<:ReactantInt}) = A
Base.ceil(A::TracedRNumber{<:ReactantFloat}) = @opcall ceil(A)
Base.ceil(A::TracedRNumber{<:ReactantInt}) = A

function Base.unsafe_trunc(
    T::Type{<:Reactant.ReactantInt}, x::TracedRNumber{<:Reactant.ReactantFloat}
)
    return @opcall convert(TracedRNumber{T}, x)
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

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.TracedRNumber}
    return (T(x.hi) + T(x.lo))::T
end

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.ConcreteRNumber}
    return Reactant.ConcreteRNumber(T(x.hi) - T(x.lo))::T
end

function Base.round(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, round(x))
end
function Base.floor(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, floor(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, ceil(x))
end

# Concatenation. Numbers in Julia are handled in a much less generic fashion than arrays
Base.vcat(x::TracedRNumber...) = Base.typed_vcat(Base.promote_eltypeof(x...), x...)
function Base.typed_vcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_vcat(T, map(Base.Fix2(Reactant.broadcast_to_size, (1,)), x)...)
end

Base.hcat(x::TracedRNumber...) = Base.typed_hcat(Base.promote_eltypeof(x...), x...)
function Base.typed_hcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_hcat(T, map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), x)...)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, xs::TracedRNumber...)
    return Base.typed_hvcat(Base.promote_eltypeof(xs...), rows, xs...)
end
function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Vararg{Int}}, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvcat(T, rows, xs...)
end

function Base.hvncat(dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...)
    return Base.typed_hvncat(Base.promote_eltypeof(xs...), dims, row_first, xs...)
end
function Base.typed_hvncat(
    ::Type{T}, dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvncat(T, dims, row_first, xs...)
end

for (Ti, Tf) in ((Int16, Float16), (Int32, Float32), (Int64, Float64))
    @eval begin
        Base.signbit(x::TracedRNumber{$(Ti)}) = x < 0
        Base.signbit(x::TracedRNumber{$(Tf)}) = signbit(@opcall(bitcast_convert($(Ti), x)))
    end
end
Base.signbit(::TracedRNumber{<:Unsigned}) = Reactant.promote_to(TracedRNumber{Bool}, false)

function Base.copysign(x::TracedRNumber, y::TracedRNumber)
    return ifelse(signbit(y), -one(x), one(x)) * abs(x)
end
function Base.copysign(x::TracedRNumber{T}, y::S) where {T,S<:Number}
    return copysign(x, Reactant.promote_to(TracedRNumber{S}, y))
end
function Base.copysign(x::S, y::TracedRNumber{T}) where {S<:Number,T}
    return copysign(Reactant.promote_to(TracedRNumber{S}, x), y)
end

function Base.zeros(::Type{TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(zero(T), dims)
end
function Base.zeros(::Type{TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(zero(T), ())
end

function Base.ones(::Type{TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(one(T), dims)
end
function Base.ones(::Type{TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(one(T), ())
end

function Base.fill(v::TracedRNumber{T}, dims::Dims{N}) where {T,N}
    return @opcall fill(v, dims)
end
function Base.fill(v::TracedRNumber{T}, ::Tuple{}) where {T}
    return @opcall fill(v, ())
end

end # module TracedRNumberOverrides
