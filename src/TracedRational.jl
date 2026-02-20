module TracedRationalOverrides

using ..Reactant: Reactant, AbstractConcreteNumber, TracedRNumber
import ..Reactant: TracedRational
using ..Ops: @opcall
using ReactantCore: ReactantCore

ReactantCore.is_traced(::TracedRational, seen) = true
ReactantCore.is_traced(::TracedRational) = true

function checked_den(::Type{T}, num, den) where {T<:Integer}
    num = Reactant.promote_to(TracedRNumber{T}, num)
    den = Reactant.promote_to(TracedRNumber{T}, den)
    den_is_neg = signbit(den)
    return TracedRational(ifelse(den_is_neg, -num, num), abs(den))
end

function checked_den(num, den)
    return checked_den(
        promote_type(Reactant.unwrapped_eltype(num), Reactant.unwrapped_eltype(den)),
        num,
        den,
    )
end

# Constructors
TracedRational{T}(num) where {T} = TracedRational(T(num), one(T))
TracedRational(num) = TracedRational(num, one(num))

# Conversion from Rational
TracedRational{T}(r::TracedRational{T}) where {T} = r
function TracedRational{T}(r::TracedRational) where {T}
    return TracedRational{T}(convert(T, r.num), convert(T, r.den))
end
function TracedRational{T}(r::Rational) where {T}
    return TracedRational{T}(convert(T, numerator(r)), convert(T, denominator(r)))
end
TracedRational(r::Rational{T}) where {T} = TracedRational{T}(numerator(r), denominator(r))

# Conversion to Rational
function Base.Rational(x::TracedRational{<:AbstractConcreteNumber})
    return Rational(Reactant.to_number(x.num), Reactant.to_number(x.den))
end

# Basic properties
Base.numerator(x::TracedRational) = x.num
Base.denominator(x::TracedRational) = x.den
function Base.denominator(::TracedRNumber{T}) where {T<:Integer}
    return Reactant.promote_to(TracedRNumber, one(T))
end

Base.show(io::IO, x::TracedRational) = print(io, x.num, " // ", x.den)

# Promotion
function Base.promote_rule(
    ::Type{TracedRational{T1}}, ::Type{TracedRational{T2}}
) where {T1,T2}
    return TracedRational{promote_type(T1, T2)}
end
function Base.promote_rule(::Type{TracedRational{T}}, ::Type{S}) where {T,S<:Integer}
    return TracedRational{promote_type(T, S)}
end
function Base.promote_rule(::Type{TracedRational{T}}, ::Type{Rational{S}}) where {T,S}
    return TracedRational{promote_type(T, S)}
end

# Operations
Base.sign(x::TracedRational) = oftype(x, sign(x.num))
Base.signbit(x::TracedRational) = signbit(x.num)
function Base.copysign(x::TracedRational, y::Real)
    return TracedRational(copysign(x.num, y), x.den)
end
function Base.copysign(x::TracedRational, y::Rational)
    return TracedRational(copysign(x.num, y.num), x.den)
end

Base.abs(x::TracedRational) = TracedRational(abs(x.num), x.den)

function Base.typemin(::Type{TracedRational{T}}) where {T<:Reactant.ReactantSInt}
    return TracedRational(-one(T), one(T))
end
function Base.typemin(::Type{TracedRational{T}}) where {T<:Reactant.ReactantUInt}
    return TracedRational(zero(T), one(T))
end
function Base.typemax(::Type{TracedRational{T}}) where {T<:Reactant.ReactantUInt}
    return TracedRational(one(T), zero(T))
end

Base.isinteger(x::TracedRational) = x.den == 1
Base.ispow2(x::TracedRational) = ispow2(x.num) & ispow2(x.den)

Base.:+(x::TracedRational) = TracedRational(+x.num, x.den)
Base.:-(x::TracedRational) = TracedRational(-x.num, x.den)
# TODO: check for overflow/underflow for integer types

for op in (:+, :-, :rem, :mod)
    @eval begin
        function Base.$(op)(x::TracedRational, y::TracedRational)
            xd, yd = Base.divgcd(promote(x.den, y.den)...)
            return TracedRational($(op)(x.num * yd, y.num * xd), x.den * yd)
        end

        function Base.$(op)(x::TracedRational, y::Union{TracedRNumber{<:Integer},Integer})
            return TracedRational($(op)(x.num, x.den * y), x.den)
        end

        function Base.$(op)(x::Union{TracedRNumber{<:Integer},Integer}, y::TracedRational)
            return TracedRational($(op)(x * y.den, y.num), y.den)
        end
    end
end

function Base.:*(x::TracedRational, y::TracedRational)
    xn, yd = Base.divgcd(promote(x.num, y.den)...)
    yn, xd = Base.divgcd(promote(y.num, x.den)...)
    return TracedRational(xn * yn, xd * yd)
end

function Base.:*(x::TracedRational, y::Union{TracedRNumber{<:Integer},Integer})
    xd, yn = Base.divgcd(promote(x.den, y)...)
    return TracedRational(x.num * yn, xd)
end

function Base.:*(x::Union{TracedRNumber{<:Integer},Integer}, y::TracedRational)
    xn, yd = Base.divgcd(promote(x, y.den)...)
    return TracedRational(xn * y.num, yd)
end

Base.inv(x::TracedRational) = TracedRational(x.den, x.num)

# Conversion operations
Base.float(x::TracedRational) = float(x.num) / float(x.den)
for T in (:(Base.Float16), :(Base.Float32), :(Base.Float64))
    @eval ($T)(x::TracedRational) =
        Reactant.promote_to(TracedRNumber{($T)}, x.num) /
        Reactant.promote_to(TracedRNumber{($T)}, x.den)
end

# Type conversion
Base.convert(::Type{TracedRational{T}}, x::TracedRational{T}) where {T} = x
Base.convert(::Type{TracedRational{T}}, x::TracedRational) where {T} = TracedRational{T}(x)
Base.convert(::Type{TracedRational{T}}, x::Rational) where {T} = TracedRational{T}(x)
Base.convert(::Type{TracedRational{T}}, x::Integer) where {T} = TracedRational{T}(x)

# Other utility functions
Base.zero(::Type{TracedRational{T}}) where {T} = TracedRational(zero(T), one(T))
Base.one(::Type{TracedRational{T}}) where {T} = TracedRational(one(T), one(T))
Base.zero(::TracedRational{T}) where {T} = zero(T)
Base.one(::TracedRational{T}) where {T} = one(T)

# Rational construction with // operator
for (T1, T2) in Iterators.product(
    (Integer, TracedRNumber{<:Integer}, AbstractConcreteNumber{<:Integer}),
    (Integer, TracedRNumber{<:Integer}, AbstractConcreteNumber{<:Integer}),
)
    T1 == T2 && T1 == Integer && continue

    @eval function Base.://(num::$(T1), den::$(T2))
        return TracedRational(
            promote(
                Reactant.promote_to(TracedRNumber, num),
                Reactant.promote_to(TracedRNumber, den),
            )...,
        )
    end
end

Base.://(num::TracedRational, den::Rational) = num//TracedRational(den)
Base.://(num::Rational, den::TracedRational) = TracedRational(num)//den

function Base.://(x::TracedRational, y::Union{TracedRNumber{<:Integer},Integer})
    xn, yn = Base.divgcd(promote(x.num, y)...)
    return checked_den(xn, x.den * yn)
end

function Base.://(x::Union{TracedRNumber{<:Integer},Integer}, y::TracedRational)
    xn, yn = Base.divgcd(promote(x, y.num)...)
    return checked_den(xn * y.den, yn)
end

function Base.://(x::TracedRational, y::TracedRational)
    xn, yn = Base.divgcd(promote(x.num, y.num)...)
    xd, yd = Base.divgcd(promote(x.den, y.den)...)
    return checked_den(xn * yd, xd * yn)
end

Base.:/(x::TracedRational, y::Union{TracedRNumber{<:Integer},Integer}) = x//y
Base.:/(x::Union{TracedRNumber{<:Integer},Integer}, y::TracedRational) = x//y
Base.:/(x::TracedRational, y::TracedRational) = x//y

end
