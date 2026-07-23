module TracedRNumberOverrides

using ..Reactant:
    Reactant,
    TracedRNumber,
    TracedRReal,
    TracedRInteger,
    TracedRFloat,
    TracedRComplex,
    TracedRArray,
    Ops,
    unwrapped_eltype,
    traced_number_type
using ..Ops: @opcall
using ReactantCore: ReactantCore, @trace
using BFloat16s: BFloat16
using Adapt: Adapt

# This isn't technically necessary in this module, but this type used to be
# defined in this module so we keep this alias here for compatibility.  TODO(#2236):
# can be removed in future breaking version of Reactant.
const TracedStepRangeLen = Reactant.TracedStepRangeLen

import Base.TwicePrecision

ReactantCore.is_traced(::TracedRNumber, seen) = true
ReactantCore.is_traced(::TracedRNumber) = true

Base.to_index(x::TracedRNumber{<:Integer}) = x

Base.precision(x::TracedRFloat{T}; kwargs...) where {T} = precision(T; kwargs...)
Base.precision(::Type{<:TracedRFloat{T}}; kwargs...) where {T} = precision(T; kwargs...)

Base.zero(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

Base.copy(x::TracedRNumber{T}) where {T} = TracedRNumber{T}((), x.mlir_data)

function Base.eps(::Type{<:TracedRFloat{T}}) where {T}
    return Reactant.promote_to(TracedRFloat{T}, eps(T))
end
Base.eps(x::TracedRFloat) = eps(typeof(x))

function Base.typemin(::Type{<:TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemin(T))
end
Base.typemin(x::TracedRNumber) = typemin(typeof(x))

function Base.typemax(::Type{<:TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemax(T))
end
Base.typemax(x::TracedRNumber) = typemax(typeof(x))

Base.floatmin(::Type{<:TracedRFloat{T}}) where {T} = floatmin(T)
Base.floatmax(::Type{<:TracedRFloat{T}}) where {T} = floatmax(T)

Base.nextfloat(x::TracedRFloat) = @opcall next_after(x, typemax(x))
Base.prevfloat(x::TracedRFloat) = @opcall next_after(x, typemin(x))

function Base.rtoldefault(T::Type{<:TracedRNumber})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end
function Base.rtoldefault(T::Type{<:TracedRFloat})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end

function Base.isfinite(x::TracedRComplex)
    return isfinite(real(x)) & isfinite(imag(x))
end
Base.isfinite(x::TracedRFloat) = @opcall is_finite(x)

function Base.isnan(x::TracedRComplex)
    return isnan(real(x)) | isnan(imag(x))
end
function Base.isnan(x::TracedRFloat{T}) where {T}
    return !isfinite(x) & (x != typemax(T)) & (x != typemin(T))
end

Base.isinf(x::TracedRComplex) = isinf(real(x)) | isinf(imag(x))
Base.isinf(x::TracedRFloat) = @opcall is_inf(x)

for TracedT in (:TracedRInteger, :TracedRFloat, :TracedRComplex)
    @eval function Base.show(io::IO, X::$TracedT{T}) where {T}
        return print(io, $(string(TracedT)), "{", T, "}(", X.paths, ")")
    end
end

Base.only(A::TracedRNumber{T}) where {T} = A

# The result of promoting the primitive types selects the traced leaf type, so
# a single parametric rule covers integer/float/complex crossings.
function Base.promote_rule(
    ::Type{<:TracedRNumber{T}}, ::Type{<:TracedRNumber{S}}
) where {T,S}
    return traced_number_type(Base.promote_type(T, S))
end

function Base.promote_rule(::Type{S}, ::Type{<:TracedRNumber{T}}) where {T,S<:Number}
    return traced_number_type(Base.promote_type(T, S))
end

function Base.promote_rule(::Type{<:TracedRNumber{T}}, ::Type{S}) where {T,S<:Number}
    return traced_number_type(Base.promote_type(T, S))
end

# Bool has special promotion rules in Base
function Base.promote_rule(::Type{Bool}, ::Type{<:TracedRNumber{T}}) where {T}
    return traced_number_type(T)
end
function Base.promote_rule(::Type{<:TracedRNumber{T}}, ::Type{Bool}) where {T}
    return traced_number_type(T)
end

# Disambiguate against Base's irrational rules on `Number` and `Real`.
function Base.promote_rule(
    T::Type{<:AbstractIrrational}, ::Type{<:TracedRNumber{S}}
) where {S}
    return traced_number_type(Base.promote_type(T, S))
end

function Base.promote_rule(
    ::Type{<:TracedRNumber{S}}, T::Type{<:AbstractIrrational}
) where {S}
    return traced_number_type(Base.promote_type(T, S))
end

function Base.promote_rule(
    T::Type{<:AbstractIrrational}, ::Type{<:TracedRReal{S}}
) where {S}
    return traced_number_type(Base.promote_type(T, S))
end

function Base.promote_rule(
    ::Type{<:TracedRReal{S}}, T::Type{<:AbstractIrrational}
) where {S}
    return traced_number_type(Base.promote_type(T, S))
end

# Disambiguate against the float8 promotion rules in PrimitiveTypes.jl, which
# apply to `Type{<:Integer}` and hence to traced integers.
for S in Base.uniontypes(Reactant.ReactantFloat8)
    @eval begin
        function Base.promote_rule(::Type{$S}, ::Type{<:TracedRInteger{T}}) where {T}
            return traced_number_type(Base.promote_type(T, $S))
        end
        function Base.promote_rule(::Type{<:TracedRInteger{T}}, ::Type{$S}) where {T}
            return traced_number_type(Base.promote_type(T, $S))
        end
    end
end

# Base defines its own `BigInt`/`BigFloat` promotion rules per numeric kind;
# these methods only disambiguate against them.
function Base.promote_rule(::Type{BigInt}, ::Type{<:TracedRInteger{T}}) where {T}
    return traced_number_type(Base.promote_type(BigInt, T))
end
function Base.promote_rule(::Type{BigInt}, ::Type{<:TracedRFloat{T}}) where {T}
    return traced_number_type(Base.promote_type(BigInt, T))
end
function Base.promote_rule(::Type{BigFloat}, ::Type{<:TracedRReal{T}}) where {T}
    return traced_number_type(Base.promote_type(BigFloat, T))
end
function Base.promote_rule(::Type{BigFloat}, ::Type{<:TracedRFloat{T}}) where {T}
    return traced_number_type(Base.promote_type(BigFloat, T))
end

# Disambiguate against Base's `Rational` and `Complex` promotion rules. Both
# argument orders are needed: the generic rules above intercept the reverse
# direction before Base's `Bottom` fallback, and their result computation
# cannot represent rationals.
function Base.promote_rule(
    ::Type{Rational{T}}, ::Type{<:TracedRInteger{S}}
) where {T<:Integer,S}
    return Reactant.TracedRational{traced_number_type(Base.promote_type(T, S))}
end
function Base.promote_rule(
    ::Type{<:TracedRInteger{S}}, ::Type{Rational{T}}
) where {T<:Integer,S}
    return Reactant.TracedRational{traced_number_type(Base.promote_type(T, S))}
end
function Base.promote_rule(
    ::Type{Rational{T}}, ::Type{<:TracedRFloat{S}}
) where {T<:Integer,S}
    return traced_number_type(Base.promote_type(Rational{T}, S))
end
function Base.promote_rule(
    ::Type{<:TracedRFloat{S}}, ::Type{Rational{T}}
) where {T<:Integer,S}
    return traced_number_type(Base.promote_type(Rational{T}, S))
end
function Base.promote_rule(::Type{Complex{T}}, ::Type{<:TracedRReal{S}}) where {T<:Real,S}
    return traced_number_type(Base.promote_type(Complex{T}, S))
end

function Base.promote_rule(::Type{Nothing}, ::Type{<:TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{<:TracedRNumber{T}}, ::Type{Nothing}) where {T}
    return Union{Nothing,TracedRNumber{T}}
end

function Base.promote_rule(::Type{Missing}, ::Type{<:TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{<:TracedRNumber{T}}, ::Type{Missing}) where {T}
    return Union{Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    ::Type{Union{Nothing,Missing}}, ::Type{<:TracedRNumber{S}}
) where {S}
    return Union{Nothing,Missing,TracedRNumber{S}}
end

function Base.promote_rule(
    ::Type{<:TracedRNumber{T}}, ::Type{Union{Nothing,Missing}}
) where {T}
    return Union{Nothing,Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    T::Type{>:Union{Nothing,Missing}}, ::Type{<:TracedRNumber{S}}
) where {S}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(
    ::Type{<:TracedRNumber{T}}, S::Type{>:Union{Nothing,Missing}}
) where {T}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(T::Type{>:Missing}, ::Type{<:TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{<:TracedRNumber{T}}, S::Type{>:Missing}) where {T}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(T, S))}}
end

function Base.promote_rule(T::Type{>:Nothing}, ::Type{<:TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{<:TracedRNumber{T}}, S::Type{>:Nothing}) where {T}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(T, S))}}
end

function Base.promote_rule(
    ::Type{TwicePrecision{T}}, ::Type{<:TracedRNumber{S}}
) where {T,S}
    return TwicePrecision{Base.promote_type(T, TracedRNumber{S})}
end

function Base.promote_rule(
    ::Type{<:TracedRNumber{T}}, ::Type{TwicePrecision{S}}
) where {T,S}
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

# Covers the concrete traced types, e.g. via `convert(TracedRFloat{Float64}, x)`.
# The extra methods disambiguate against Base's constructors on `Real`,
# `AbstractFloat`, and `Integer`.
(::Type{RT})(x::Number) where {RT<:TracedRNumber} = Reactant.promote_to(RT, x)
for (RT, XT) in (
    (:TracedRReal, :Complex),
    (:TracedRInteger, :Complex),
    (:TracedRFloat, :Complex),
    (:TracedRInteger, :Rational),
    (:TracedRInteger, :BigFloat),
)
    @eval (::Type{RT})(x::$XT) where {RT<:$RT} = Reactant.promote_to(RT, x)
end
# The ambiguity resolver requires the exact TypeVar shape of the competing
# Base constructor: `(::Type{T})(::Rational{S}) where {S,T<:AbstractFloat}`
# is `S`-parameterized, `(::Type{T})(::Rational) where {T<:Integer}` is not.
(::Type{RT})(x::Rational{S}) where {S,RT<:TracedRFloat} = Reactant.promote_to(RT, x)

for T in Base.uniontypes(Reactant.ReactantFloat8)
    @eval TracedRNumber{T}(x::$T) where {T} = Reactant.promote_to(TracedRNumber{T}, x)
end

for (aT, bT) in ((TracedRReal, Real), (Real, TracedRReal), (TracedRReal, TracedRReal))
    @eval function Base.Complex(a::$aT, b::$bT)
        T = promote_type(unwrapped_eltype(a), unwrapped_eltype(b))
        a = Reactant.promote_to(TracedRNumber{T}, a)
        b = Reactant.promote_to(TracedRNumber{T}, b)
        return @opcall complex(a, b)
    end
end

Base.Complex(x::TracedRReal) = @opcall complex(x, zero(x))
Base.Complex(x::TracedRComplex) = x

# Base.complex
Base.complex(::Type{TracedRNumber{T}}) where {T} = traced_number_type(complex(T))
Base.complex(::Type{<:TracedRReal{T}}) where {T} = traced_number_type(complex(T))
Base.complex(::Type{<:TracedRComplex{T}}) where {T} = TracedRComplex{T}
Base.complex(x::TracedRReal) = complex(x, zero(x))
function Base.complex(x::TracedRReal, y::TracedRReal)
    T = promote_type(unwrapped_eltype(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRReal, y::Real)
    T = promote_type(unwrapped_eltype(x), typeof(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::Real, y::TracedRReal)
    T = promote_type(typeof(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRReal{T}, y::TracedRReal{T}) where {T<:Real}
    return @opcall complex(x, y)
end
Base.complex(x::TracedRComplex) = x

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
    (:(Base.:*), :multiply),
    (:(Base.:/), :divide),
    (:(Base.:^), :power),
)
    @eval function $(jlop)(lhs::TracedRNumber{T}, rhs::TracedRNumber{T}) where {T}
        return @opcall $(hloop)(lhs, rhs)
    end
end

# real-only, so that it dominates the mixed-argument methods below
function Base.rem(lhs::TracedRReal{T}, rhs::TracedRReal{T}) where {T}
    return @opcall remainder(lhs, rhs)
end

function Base.rem(x::TracedRReal, y::TracedRReal, ::typeof(Base.RoundFromZero))
    return ifelse(
        signbit(x) == signbit(y), rem(x, y, Base.RoundUp), rem(x, y, Base.RoundDown)
    )
end

function Base.:*(x::TracedRReal, z::Complex{Bool})
    # this is to support multiplication by im (Complex{Bool}(false, true))
    z_re, z_im = real(z), imag(z)
    res_re = z_re ? x : zero(x)
    res_im = z_im ? x : zero(x)
    return Complex(res_re, res_im)
end
Base.:*(z::Complex{Bool}, x::TracedRReal) = x * z

# Based on https://github.com/JuliaLang/julia/blob/39255d47db7657950ff1c82137ecec5a70bae622/base/float.jl#L608-L617
function Base.mod(
    @nospecialize(x::TracedRReal{T}), @nospecialize(y::TracedRReal{T})
) where {T}
    r = rem(x, y)
    return ifelse(r == 0, copysign(r, y), ifelse((r > 0) ⊻ (y > 0), r + y, r))
end

function Base.mod1(
    @nospecialize(x::TracedRReal{T}), @nospecialize(y::TracedRReal{T})
) where {T}
    m = mod(x, y)
    return ifelse(m == 0, y, m)
end

for op in (:mod, :mod1, :rem)
    @eval begin
        function Base.$op(
            @nospecialize(lhs::TracedRReal{T}), @nospecialize(rhs::Real)
        ) where {T}
            return $(op)(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
        end
        function Base.$op(
            @nospecialize(lhs::Real), @nospecialize(rhs::TracedRReal{T})
        ) where {T}
            return $(op)(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
        end
    end
end

for op in (:mod, :rem)
    @eval begin
        function Base.$op(
            @nospecialize(lhs::TracedRInteger{T}), @nospecialize(rhs::Rational)
        ) where {T}
            return $(op)(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
        end
        function Base.$op(
            @nospecialize(lhs::Rational), @nospecialize(rhs::TracedRInteger{T})
        ) where {T}
            return $(op)(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
        end
    end
end

Base.flipsign(x::TracedRReal, y::TracedRReal) = ifelse(y < 0, -x, x)

function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt}, y::TracedRNumber{<:Reactant.ReactantUInt}
)
    return div(x, y, RoundDown)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundToZero),
)
    return flipsign(signed(div(unsigned(abs(x)), y)), x)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundDown),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    result = flipsign(q, x)
    return ifelse(signbit(x) & has_rem, result - one(result), result)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundUp),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    result = flipsign(q, x)
    return ifelse(!signbit(x) & has_rem, result + one(result), result)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundFromZero),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    q_adj = q + ifelse(has_rem, one(q), zero(q))
    return flipsign(q_adj, x)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantUInt}, y::TracedRNumber{<:Reactant.ReactantSInt}
)
    return unsigned(flipsign(signed(div(x, unsigned(abs(y)))), y))
end

function Base.div(
    @nospecialize(lhs::TracedRInteger{T}), rhs::Real, r::Base.RoundingMode
) where {T}
    return div(lhs, Reactant.promote_to(TracedRNumber{T}, rhs), r)
end
function Base.div(
    lhs::Real, @nospecialize(rhs::TracedRInteger{T}), r::Base.RoundingMode
) where {T}
    return div(Reactant.promote_to(TracedRNumber{T}, lhs), rhs, r)
end
function Base.div(
    @nospecialize(lhs::TracedRInteger{T1}),
    @nospecialize(rhs::TracedRInteger{T2}),
    r::Base.RoundingMode,
) where {T1,T2}
    T = promote_type(T1, T2)
    return div(
        Reactant.promote_to(TracedRNumber{T}, lhs),
        Reactant.promote_to(TracedRNumber{T}, rhs),
        r,
    )
end

function Base.div(@nospecialize(lhs::TracedRReal{T}), rhs::Real) where {T}
    return @opcall divide(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
end
function Base.div(lhs::Real, @nospecialize(rhs::TracedRReal{T})) where {T}
    return @opcall divide(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRReal{T1}), @nospecialize(rhs::TracedRReal{T2})
) where {T1,T2}
    T = promote_type(T1, T2)
    return @opcall divide(
        Reactant.promote_to(TracedRNumber{T}, lhs),
        Reactant.promote_to(TracedRNumber{T}, rhs),
    )
end

function Base.div(
    @nospecialize(lhs::TracedRInteger{T}),
    @nospecialize(rhs::TracedRInteger{T}),
    ::typeof(RoundToZero),
) where {T<:Integer}
    return @opcall divide(lhs, rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRInteger{T}),
    @nospecialize(rhs::TracedRInteger{T}),
    ::typeof(RoundDown),
) where {T<:Integer}
    return @opcall divide(lhs, rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRInteger{T}),
    @nospecialize(rhs::TracedRInteger{T}),
    ::typeof(RoundUp),
) where {T<:Integer}
    q = div(lhs, rhs)  # truncation (RoundToZero)
    return q + (!iszero(rem(lhs, rhs)) & (signbit(lhs) == signbit(rhs)))
end
function Base.div(
    @nospecialize(lhs::TracedRInteger{T}),
    @nospecialize(rhs::TracedRInteger{T}),
    ::typeof(RoundFromZero),
) where {T<:Integer}
    return ifelse(
        signbit(lhs) == signbit(rhs), div(lhs, rhs, RoundUp), div(lhs, rhs, RoundDown)
    )
end
for RM in
    (RoundingMode{:Nearest}, RoundingMode{:NearestTiesAway}, RoundingMode{:NearestTiesUp})
    @eval function Base.div(
        @nospecialize(lhs::TracedRInteger{T}), @nospecialize(rhs::TracedRInteger{T}), r::$RM
    ) where {T<:Integer}
        return divrem(lhs, rhs, r)[1]
    end
end

function Base.div(
    @nospecialize(x::TracedRFloat{T}), @nospecialize(y::TracedRFloat{T}), r::RoundingMode
) where {T}
    return round(div(x, y), r)
end

Base.div(@nospecialize(lhs::TracedRInteger), ::Missing, ::RoundingMode) = missing
Base.div(::Missing, @nospecialize(rhs::TracedRInteger), ::RoundingMode) = missing

# Base has `div` methods specialized on single rounding modes (and on
# `Rational` arguments); the following methods only disambiguate against them.
for RM in Reactant.BASE_SPECIFIC_ROUNDING_MODES
    @eval begin
        function Base.div(
            @nospecialize(lhs::TracedRInteger{T1}),
            @nospecialize(rhs::TracedRInteger{T2}),
            r::$RM,
        ) where {T1,T2}
            T = promote_type(T1, T2)
            return div(
                Reactant.promote_to(TracedRNumber{T}, lhs),
                Reactant.promote_to(TracedRNumber{T}, rhs),
                r,
            )
        end
        function Base.div(
            @nospecialize(lhs::TracedRInteger{T}), rhs::Integer, r::$RM
        ) where {T}
            return div(lhs, Reactant.promote_to(TracedRNumber{T}, rhs), r)
        end
        function Base.div(
            lhs::Integer, @nospecialize(rhs::TracedRInteger{T}), r::$RM
        ) where {T}
            return div(Reactant.promote_to(TracedRNumber{T}, lhs), rhs, r)
        end
    end
end
function Base.div(
    @nospecialize(lhs::TracedRInteger{T}), rhs::Rational, r::RoundingMode
) where {T}
    return div(lhs, Reactant.promote_to(TracedRNumber{T}, rhs), r)
end
function Base.div(
    lhs::Rational, @nospecialize(rhs::TracedRInteger{T}), r::RoundingMode
) where {T}
    return div(Reactant.promote_to(TracedRNumber{T}, lhs), rhs, r)
end
function Base.div(@nospecialize(lhs::TracedRInteger{T}), rhs::Rational) where {T}
    return div(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
end
function Base.div(lhs::Rational, @nospecialize(rhs::TracedRInteger{T})) where {T}
    return div(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
end

function Base.divrem(
    @nospecialize(a::TracedRInteger{T}),
    @nospecialize(b::TracedRInteger{T}),
    r::Union{typeof(RoundUp),typeof(RoundDown),typeof(RoundToZero)},
) where {T<:Integer}
    if r === RoundToZero
        d = div(a, b)
        return (d, a - d * b)
    elseif r === RoundDown
        d = fld(a, b)
        return (d, a - d * b)
    elseif r === RoundUp
        d = div(a, b, r)
        return (d, a - d * b)
    end
end

function Base.divrem(
    @nospecialize(x::TracedRInteger{T}),
    @nospecialize(y::TracedRInteger{T}),
    ::typeof(RoundNearest),
) where {T}
    (q, r) = divrem(x, y)
    threshold = isodd(y) | iseven(q)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + threshold, (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + threshold, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - threshold, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - threshold, (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRInteger{T}),
    @nospecialize(y::TracedRInteger{T}),
    ::typeof(RoundNearestTiesAway),
) where {T}
    (q, r) = divrem(x, y)
    threshold = isodd(y)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + threshold, (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + threshold, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - threshold, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - threshold, (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRInteger{T}),
    @nospecialize(y::TracedRInteger{T}),
    ::typeof(RoundNearestTiesUp),
) where {T}
    (q, r) = divrem(x, y)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + isodd(y), (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + true, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - true, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - isodd(y), (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRInteger{T}),
    @nospecialize(y::TracedRInteger{T}),
    ::typeof(RoundFromZero),
) where {T}
    q_up, r_up = divrem(x, y, RoundUp)
    q_down, r_down = divrem(x, y, RoundDown)
    return ifelse(signbit(x) == signbit(y), (q_up, r_up), (q_down, r_down))
end

function Base.:/(
    @nospecialize(lhs::TracedRInteger{T}), @nospecialize(rhs::TracedRInteger{T})
) where {T}
    return float(lhs) / float(rhs)
end

# Comparisons are defined on the concrete traced types (not the `TracedRNumber`
# union) so that they are strictly more specific than the corresponding Base
# methods on `Real`/`AbstractFloat`/`Integer`. Order comparisons are real-only,
# matching Base; `==`/`!=` also cover traced complex numbers. Mixed-type
# arguments are promoted to a common traced type. The extra methods for
# `Rational`, `BigInt`, `BigFloat`, `AbstractIrrational`, `AbstractFloat`, and
# `Complex` resolve ambiguities with Base's specialized comparison methods.
function promote_to_common(
    @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
) where {T1,T2}
    commonTy = TracedRNumber{Base.promote_type(T1, T2)}
    return Reactant.promote_to(commonTy, lhs), Reactant.promote_to(commonTy, rhs)
end
function promote_to_common(@nospecialize(lhs::TracedRNumber), @nospecialize(rhs))
    return lhs, Reactant.promote_to(lhs, rhs)
end
function promote_to_common(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber))
    return Reactant.promote_to(rhs, lhs), rhs
end

for (jlop, hlocomp, TracedTs) in (
    (:(Base.:(==)), "EQ", (:TracedRInteger, :TracedRFloat, :TracedRComplex)),
    (:(Base.:(!=)), "NE", (:TracedRInteger, :TracedRFloat, :TracedRComplex)),
    (:(Base.:(>=)), "GE", (:TracedRInteger, :TracedRFloat)),
    (:(Base.:(>)), "GT", (:TracedRInteger, :TracedRFloat)),
    (:(Base.:(<=)), "LE", (:TracedRInteger, :TracedRFloat)),
    (:(Base.:(<)), "LT", (:TracedRInteger, :TracedRFloat)),
    (:(Base.isless), "LT", (:TracedRInteger, :TracedRFloat)),
)
    OtherT = length(TracedTs) == 2 ? :Real : :Number
    disambTs = (:Rational, :BigInt, :BigFloat, :AbstractIrrational, :AbstractFloat)
    length(TracedTs) == 3 && (disambTs = (disambTs..., :Complex))

    for T in TracedTs
        @eval begin
            function $(jlop)(@nospecialize(lhs::$T{T}), @nospecialize(rhs::$T{T})) where {T}
                return @opcall compare(lhs, rhs; comparison_direction=($(hlocomp)))
            end
            function $(jlop)(@nospecialize(lhs::$T), @nospecialize(rhs::$OtherT))
                return $(jlop)(promote_to_common(lhs, rhs)...)
            end
            function $(jlop)(@nospecialize(lhs::$OtherT), @nospecialize(rhs::$T))
                return $(jlop)(promote_to_common(lhs, rhs)...)
            end
        end
        for X in disambTs
            @eval begin
                function $(jlop)(@nospecialize(lhs::$T), @nospecialize(rhs::$X))
                    return $(jlop)(promote_to_common(lhs, rhs)...)
                end
                function $(jlop)(@nospecialize(lhs::$X), @nospecialize(rhs::$T))
                    return $(jlop)(promote_to_common(lhs, rhs)...)
                end
            end
        end
    end
    for T1 in TracedTs, T2 in TracedTs
        @eval function $(jlop)(@nospecialize(lhs::$T1), @nospecialize(rhs::$T2))
            return $(jlop)(promote_to_common(lhs, rhs)...)
        end
    end

    @eval begin
        # ambiguity fixes
        $(jlop)(@nospecialize(lhs::TracedRNumber), @nospecialize(::Missing)) = missing
        $(jlop)(@nospecialize(::Missing), @nospecialize(rhs::TracedRNumber)) = missing
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
        return Base.ifelse(pred, x[i], y[i])
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

function Base.:*(x::TwicePrecision{<:Union{Float16,Float32,Float64}}, v::TracedRInteger)
    return invoke(*, Tuple{TwicePrecision,TracedRNumber}, x, v)
end

function Base.:*(x::TwicePrecision, v::TracedRNumber)
    @trace result = if v == 0
        TwicePrecision(x.hi * v, x.lo * v)
    else
        x * TwicePrecision(oftype(x.hi * v, v))
    end
    return result
end

for (jlop, hloop) in ((:(Base.:&), :and), (:(Base.:|), :or), (:(Base.xor), :xor))
    @eval begin
        function $jlop(x::TracedRInteger{A}, y::TracedRInteger{B}) where {A,B}
            C = promote_type(A, B)
            return @opcall $hloop(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function $jlop(x::TracedRInteger{A}, y::Integer) where {A}
            C = promote_type(A, unwrapped_eltype(y))
            return $jlop(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function $jlop(x::Integer, y::TracedRInteger{B}) where {B}
            C = promote_type(unwrapped_eltype(x), B)
            return $jlop(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
    end
end
Base.:!(x::TracedRInteger) = @opcall not(x)

# With a traced integer exponent, Base's `^(::Number, ::Integer)` would call
# `power_by_squaring`, which branches on traced booleans. The extra methods
# disambiguate against Base's specialized `^` methods.
for B in (:TracedRInteger, :TracedRFloat, :TracedRComplex, :Real, :Complex)
    @eval Base.:^(x::$B, p::TracedRInteger) = ^(promote(x, p)...)
end
# A traced exponent's sign is unknown at trace time, so rationality cannot be
# preserved (`x^p` vs `inv(x)^-p`); compute in floating point instead.
Base.:^(x::Rational, p::TracedRInteger) = ^(float(x), p)
for B in (
    Float16,
    Float32,
    Union{Float16,Float32},
    Float64,
    BFloat16,
    Complex{<:AbstractFloat},
    Complex{<:Integer},
    Complex{<:Rational},
    Irrational{:ℯ},
    BigInt,
    BigFloat,
)
    @eval Base.:^(x::$B, p::TracedRInteger) = ^(promote(x, p)...)
end

# Same-`T` disambiguators against the same-type traced `^` method.
Base.:^(x::TracedRInteger{T}, p::TracedRInteger{T}) where {T} = @opcall power(x, p)
Base.:^(x::TracedRFloat{T}, p::TracedRInteger{T}) where {T} = ^(promote(x, p)...)
Base.:^(x::TracedRComplex{T}, p::TracedRInteger{T}) where {T} = ^(promote(x, p)...)
Base.:^(x::TracedRReal{T}, p::TracedRInteger{T}) where {T} = ^(promote(x, p)...)

Base.fma(x::TracedRFloat{T}, y::TracedRFloat{T}, z::TracedRFloat{T}) where {T} = x * y + z

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRNumber{T}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
    (:(Base.abs), :abs),
    (:(Base.:-), :negate),
    (:(Base.sin), :sine),
    (:(Base.sinh), :sinh),
    (:(Base.cos), :cosine),
    (:(Base.cosh), :cosh),
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
)
    # Separate real and complex methods: the union would be ambiguous with
    # Base's methods on `Real`.
    @eval $(jlop)(@nospecialize(lhs::TracedRReal)) = @opcall $(hloop)(lhs)
    @eval $(jlop)(@nospecialize(lhs::TracedRComplex)) = @opcall $(hloop)(lhs)
end

# `conj`, `real`, and `imag` of traced reals are covered by Base's identities.
for (jlop, hloop) in ((:(Base.conj), :conj), (:(Base.real), :real), (:(Base.imag), :imag))
    @eval $(jlop)(@nospecialize(lhs::TracedRComplex)) = @opcall $(hloop)(lhs)
end

# Degree-based trigonometric wrappers for TracedRNumber
# These convert to radians internally so Reactant can lower to
# StableHLO-supported radian trigonometric operations.

Base.sind(x::TracedRReal) = sin(deg2rad(x))
Base.cosd(x::TracedRReal) = cos(deg2rad(x))
Base.tand(x::TracedRReal) = tan(deg2rad(x))
Base.cscd(x::TracedRReal) = 1 / sind(x)
Base.secd(x::TracedRReal) = 1 / cosd(x)
Base.cotd(x::TracedRReal) = 1 / tand(x)

Base.asind(x::TracedRReal) = rad2deg(asin(x))
Base.acosd(x::TracedRReal) = rad2deg(acos(x))
Base.atand(x::TracedRReal) = rad2deg(atan(x))

Base.atan(y::TracedRReal{T}, x::TracedRReal{T}) where {T} = @opcall atan2(y, x)
Base.atan(y::TracedRReal, x::TracedRReal) = atan(promote_to_common(y, x)...)
Base.atand(y::TracedRReal, x::TracedRReal) = rad2deg(atan(y, x))

Base.hypot(x::TracedRReal{T}, y::TracedRReal{T}) where {T} = @opcall hypot(x, y)

function Base.hypot(x::TracedRReal, y::Real)
    return Base.hypot(promote_to_common(x, y)...)
end

function Base.hypot(x::Real, y::TracedRReal)
    return Base.hypot(promote_to_common(x, y)...)
end

function Base.hypot(x::TracedRReal, y::TracedRReal)
    return Base.hypot(promote_to_common(x, y)...)
end

Base.hypot(::Missing, ::TracedRNumber) = missing
Base.hypot(::TracedRNumber, ::Missing) = missing

Base.acscd(x::TracedRReal) = rad2deg(asin(1 / x))
Base.asecd(x::TracedRReal) = rad2deg(acos(1 / x))
Base.acotd(x::TracedRReal) = rad2deg(atan(1 / x))

for (jlop, hloop) in (
    (:(Base.sin), :sine),
    (:(Base.sinh), :sinh),
    (:(Base.cos), :cosine),
    (:(Base.cosh), :cosh),
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
    @eval $(jlop)(@nospecialize(lhs::TracedRInteger)) = @opcall $(hloop)(float(lhs))
end

for (jlop, hloop) in
    ((:(Base.sinpi), :sine), (:(Base.cospi), :cosine), (:(Base.tanpi), :tan))
    for TracedT in (:TracedRReal, :TracedRComplex)
        @eval $(jlop)(@nospecialize(lhs::$TracedT{T})) where {T} =
            @opcall $(hloop)(T(π) * lhs)
    end
end

function Base.sincospi(x::TracedRNumber{T}) where {T}
    return @opcall(sine(T(π) * x)), @opcall(cosine(T(π) * x))
end

function Base.cispi(x::TracedRReal)
    s, c = sincospi(x)
    return complex(c, s)
end

function Base.cis(x::TracedRReal)
    s, c = sincos(x)
    return complex(c, s)
end

Base.sincos(x::TracedRFloat) = (sin(x), cos(x))
Base.exp2(x::TracedRFloat{T}) where {T} = T(2)^x
Base.exp10(x::TracedRFloat{T}) where {T} = T(10)^x
Base.mod2pi(x::TracedRFloat{T}) where {T} = mod(x, T(2π))
function Base.modf(x::TracedRFloat)
    ipart = trunc(x)
    return (x - ipart, ipart)
end

function Base.sinc(x::TracedRNumber)
    r = ifelse(iszero(x), one(x), sinpi(x) / (pi * x))
    return ifelse(isinf(x), zero(x), r)
end

function Base.sinc(x::TracedRComplex{Complex{T}}) where {T}
    r1 = ifelse(
        abs(x) < Base.Math._sinc_threshold(T),
        evalpoly(x^2, (T(1), -T(pi)^2 / 6, T(pi)^4 / 120)),
        sinpi(x) / (pi * x),
    )
    return ifelse(isinf(real(x)), zero(x), r1)
end

for TracedT in (:TracedRReal, :TracedRComplex)
    @eval begin
        @noinline Base.Math.log10(x::$TracedT) = Base.Math._log(x, Val(10), :log10)
        @noinline Base.Math.log2(x::$TracedT) = Base.Math._log(x, Val(2), :log2)
    end
end
Base.Math._log(x::TracedRNumber, base, ::Symbol) = log(x) / log(Reactant._unwrap_val(base))

Base.isreal(x::TracedRComplex) = iszero(imag(x))

Base.isodd(x::TracedRComplex) = isodd(real(x))
Base.isodd(x::TracedRInteger) = !iszero(rem(x, 2))
function Base.isodd(x::TracedRFloat)
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

Base.iseven(x::TracedRComplex) = iseven(real(x))
Base.iseven(x::TracedRInteger) = iszero(rem(x, 2))
function Base.iseven(x::TracedRFloat)
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

Base.float(x::TracedRFloat) = x
function Base.float(x::Union{TracedRInteger{T},TracedRComplex{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{float(T)}, x)
end

Base.float(::Type{TracedRNumber{T}}) where {T} = traced_number_type(float(T))
for TracedT in (:TracedRInteger, :TracedRFloat, :TracedRComplex)
    @eval Base.float(::Type{<:$TracedT{T}}) where {T} = traced_number_type(float(T))
end

Base.round(A::TracedRFloat) = @opcall round_nearest_even(A)
Base.round(A::TracedRInteger) = A
function Base.round(A::TracedRFloat, ::typeof(RoundNearest))
    return @opcall round_nearest_even(A)
end
function Base.round(A::TracedRFloat, ::typeof(RoundNearestTiesAway))
    return @opcall round_nearest_afz(A)
end
Base.round(A::TracedRFloat, ::typeof(RoundUp)) = ceil(A)
Base.round(A::TracedRFloat, ::typeof(RoundDown)) = floor(A)
function Base.round(A::TracedRFloat, ::typeof(RoundToZero))
    A_truncated = @opcall convert(TracedRNumber{Int}, A)
    return copysign(
        @opcall(convert(TracedRNumber{Reactant.unwrapped_eltype(A)}, A_truncated)), A
    )
end

Base.floor(A::TracedRFloat) = @opcall floor(A)
Base.floor(A::TracedRInteger) = A
Base.ceil(A::TracedRFloat) = @opcall ceil(A)
Base.ceil(A::TracedRInteger) = A

function Base.unsafe_trunc(T::Type{<:Reactant.ReactantInt}, x::TracedRFloat)
    return @opcall convert(TracedRNumber{T}, x)
end

for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for Tf in (Float16, Float32, Float64)
        if Ti <: Unsigned || sizeof(Ti) < sizeof(Tf)
            # Here `Tf(typemin(Ti))-1` is exact, so we can compare the lower-bound
            # directly. `Tf(typemax(Ti))+1` is either always exactly representable, or
            # rounded to `Inf` (e.g. when `Ti==UInt128 && Tf==Float32`).
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRFloat{$Tf})
                    # TODO(#2236) throw error within traced
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
                function Base.trunc(::Type{$Ti}, x::TracedRFloat{$Tf})
                    # TODO(#2236) throw error within traced
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
Base.trunc(::Type{Signed}, x::TracedRFloat{<:Base.IEEEFloat}) = Base.trunc(Int, x)
Base.trunc(::Type{Unsigned}, x::TracedRFloat{<:Base.IEEEFloat}) = Base.trunc(UInt, x)
Base.trunc(::Type{Integer}, x::TracedRFloat{<:Base.IEEEFloat}) = Base.trunc(Int, x)

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.TracedRNumber}
    return (T(x.hi) + T(x.lo))::T
end

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.ConcreteRNumber}
    return Reactant.ConcreteRNumber(T(x.hi) - T(x.lo))::T
end

function Base.round(::Type{T}, x::TracedRFloat) where {T<:Integer}
    return trunc(T, round(x))
end
function Base.floor(::Type{T}, x::TracedRFloat) where {T<:Integer}
    return trunc(T, floor(x))
end
function Base.ceil(::Type{T}, x::TracedRFloat) where {T<:Integer}
    return trunc(T, ceil(x))
end

# disambiguate against 1.10's `round/floor/ceil(::Type{Bool}, ::AbstractFloat)`
Base.round(::Type{Bool}, x::TracedRFloat) = trunc(Bool, round(x))
Base.floor(::Type{Bool}, x::TracedRFloat) = trunc(Bool, floor(x))
Base.ceil(::Type{Bool}, x::TracedRFloat) = trunc(Bool, ceil(x))

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

Base.signbit(x::TracedRInteger{<:Reactant.ReactantSInt}) = x < 0
function Base.signbit(::TracedRInteger{<:Union{Bool,Reactant.ReactantUInt}})
    return Reactant.promote_to(TracedRNumber{Bool}, false)
end
for (Ti, Tf) in ((Int16, Float16), (Int32, Float32), (Int64, Float64))
    @eval Base.signbit(x::TracedRFloat{$(Tf)}) = signbit(@opcall(bitcast_convert($(Ti), x)))
end

function Base.copysign(x::TracedRReal, y::TracedRReal)
    return ifelse(signbit(y), -one(x), one(x)) * abs(x)
end
function Base.copysign(x::TracedRReal, y::S) where {S<:Real}
    return copysign(x, Reactant.promote_to(TracedRNumber{unwrapped_eltype(S)}, y))
end
function Base.copysign(x::S, y::TracedRReal) where {S<:Real}
    return copysign(Reactant.promote_to(TracedRNumber{unwrapped_eltype(S)}, x), y)
end

# Base specializes `copysign` on these first-argument types.
for S in (:Float16, :Float32, :Float64, :Signed, :Rational)
    @eval function Base.copysign(x::$S, y::TracedRReal)
        return copysign(Reactant.promote_to(TracedRNumber{unwrapped_eltype(x)}, x), y)
    end
end

function Base.zeros(::Type{<:TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(zero(T), dims)
end
function Base.zeros(::Type{<:TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(zero(T), ())
end

function Base.ones(::Type{<:TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(one(T), dims)
end
function Base.ones(::Type{<:TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(one(T), ())
end

function Base.fill(v::TracedRNumber{T}, dims::Dims{N}) where {T,N}
    return @opcall fill(v, dims)
end
function Base.fill(v::TracedRNumber{T}, ::Tuple{}) where {T}
    return @opcall fill(v, ())
end

# TODO(#2236): actually perform bounds checking
function Base.checkindex(::Type{Bool}, _inds, ::TracedRNumber)
    @warn "Currently we don't perform bounds checking for TracedRNumber. This will be \
           fixed in a future version of Reactant." maxlog = 1
    return true
end

function Base.checkindex(::Type{Bool}, ::AbstractUnitRange, ::TracedRNumber)
    @warn "Currently we don't perform bounds checking for TracedRNumber. This will be \
           fixed in a future version of Reactant." maxlog = 1
    return true
end

function Base.checkindex(::Type{Bool}, ::Base.IdentityUnitRange, ::TracedRReal)
    @warn "Currently we don't perform bounds checking for TracedRNumber. This will be \
           fixed in a future version of Reactant." maxlog = 1
    return true
end

# rem2pi: fallback to rem for now.
# TODO(#2259): we should replace with the more numerically stable version.
# https://github.com/JuliaLang/julia/blob/4d04bb6b3b1b879f4dbb918d194c5c939a1e7f3c/base/special/rem2pi.jl#L133
Base.rem2pi(x::TracedRReal, r::Base.RoundingMode) = rem(x, typeof(x)(2π), r)

function Base.rem2pi(
    x::T, r::Base.RoundingMode
) where {T<:TracedRFloat{<:Union{Float16,Float32}}}
    return T(rem2pi(TracedRNumber{Float64}(x), r))
end
Base.rem2pi(x::TracedRInteger, r::Base.RoundingMode) = rem2pi(float(x), r)

@static if isdefined(Base, :unchecked_oneto)
    function Base.unchecked_oneto(x::TracedRInteger)
        return Reactant.TracedUnitRange(one(x), x)
    end
end

Base.unsigned(x::TracedRInteger{T}) where {T<:Reactant.ReactantUInt} = x
function Base.unsigned(x::TracedRInteger{T}) where {T<:Reactant.ReactantSInt}
    return convert(TracedRNumber{unsigned(T)}, x)
end

function Base.signed(x::TracedRInteger{T}) where {T<:Reactant.ReactantUInt}
    return convert(TracedRNumber{signed(T)}, x)
end
Base.signed(x::TracedRInteger{T}) where {T<:Reactant.ReactantSInt} = x

Base.uabs(x::TracedRInteger) = abs(x)
Base.uabs(x::TracedRInteger{<:Reactant.ReactantSInt}) = unsigned(abs(x))

function Base.divgcd(x::TracedRInteger, y::TracedRInteger)
    g = gcd(Base.uabs(x), Base.uabs(y))
    return div(x, g), div(y, g)
end

# See https://github.com/EnzymeAD/Reactant.jl/issues/2476 for why this is
# implemented this way.
function Base.gcd(a::TracedRInteger{T}, b::TracedRInteger{T}) where {T}
    a_tmp = copy(a)
    b_tmp = copy(b)
    @trace while b_tmp != 0
        t = b_tmp
        b_tmp = rem(a_tmp, b_tmp)
        a_tmp = t
    end
    return abs(a_tmp)
end

end # module TracedRNumberOverrides
