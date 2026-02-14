module TracedRationalOverrides

using ..Reactant: Reactant, AbstractConcreteNumber, TracedRNumber
import ..Reactant: TracedRational
using ..Ops: @opcall
using ReactantCore: ReactantCore

ReactantCore.is_traced(::TracedRational, seen) = true
ReactantCore.is_traced(::TracedRational) = true

# Constructors
function TracedRational{T}(num::T, den::T) where {T}
    return TracedRational{T}(num, den)
end
function TracedRational{T}(num, den) where {T}
    return TracedRational{T}(convert(T, num), convert(T, den))
end
function TracedRational(num::T, den::T) where {T}
    return TracedRational{T}(num, den)
end
function TracedRational(num, den)
    num_den_promoted = promote(num, den)
    return TracedRational(num_den_promoted...)
end
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

function Base.show(io::IO, x::TracedRational{T}) where {T}
    return print(io, "TracedRational{", T, "}(", x.num, " // ", x.den, ")")
end

# Promotion
function Base.promote_rule(
    ::Type{TracedRational{T1}}, ::Type{TracedRational{T2}}
) where {T1,T2}
    T = promote_type(T1, T2)
    return TracedRational{T}
end
function Base.promote_rule(::Type{TracedRational{T}}, ::Type{<:Integer}) where {T}
    return TracedRational{T}
end
function Base.promote_rule(
    ::Type{TracedRational{T}}, ::Type{Rational{S}}
) where {T,S}
    return TracedRational{promote_type(T, S)}
end

# Arithmetic operations
function Base.:+(x::TracedRational, y::TracedRational)
    num = @opcall +(x.num * y.den, y.num * x.den)
    den = @opcall *(x.den, y.den)
    return TracedRational(num, den)
end

function Base.:-(x::TracedRational, y::TracedRational)
    num = @opcall -(x.num * y.den, y.num * x.den)
    den = @opcall *(x.den, y.den)
    return TracedRational(num, den)
end

function Base.:*(x::TracedRational, y::TracedRational)
    num = @opcall *(x.num, y.num)
    den = @opcall *(x.den, y.den)
    return TracedRational(num, den)
end

function Base.:/(x::TracedRational, y::TracedRational)
    num = @opcall *(x.num, y.den)
    den = @opcall *(x.den, y.num)
    return TracedRational(num, den)
end

function Base.:-(x::TracedRational)
    return TracedRational(@opcall -(x.num), x.den)
end

function Base.:+(x::TracedRational, y::Integer)
    return TracedRational(@opcall +(x.num, x.den * y), x.den)
end
Base.:+(y::Integer, x::TracedRational) = x + y

function Base.:+(x::TracedRational, y::TracedRNumber)
    return TracedRational(@opcall +(x.num, x.den * y), x.den)
end
Base.:+(y::TracedRNumber, x::TracedRational) = x + y

function Base.:-(x::TracedRational, y::Integer)
    return TracedRational(@opcall -(x.num, x.den * y), x.den)
end
function Base.:-(y::Integer, x::TracedRational)
    return TracedRational(@opcall -(x.den * y, x.num), x.den)
end

function Base.:-(x::TracedRational, y::TracedRNumber)
    return TracedRational(@opcall -(x.num, x.den * y), x.den)
end
function Base.:-(y::TracedRNumber, x::TracedRational)
    return TracedRational(@opcall -(x.den * y, x.num), x.den)
end

function Base.:*(x::TracedRational, y::Integer)
    return TracedRational(@opcall *(x.num, y), x.den)
end
Base.:*(y::Integer, x::TracedRational) = x * y

function Base.:*(x::TracedRational, y::TracedRNumber)
    return TracedRational(@opcall *(x.num, y), x.den)
end
Base.:*(y::TracedRNumber, x::TracedRational) = x * y

function Base.:/(x::TracedRational, y::Integer)
    return TracedRational(x.num, @opcall *(x.den, y))
end
function Base.:/(x::Integer, y::TracedRational)
    return TracedRational(@opcall *(x, y.den), y.num)
end

function Base.:/(x::TracedRational, y::TracedRNumber)
    return TracedRational(x.num, @opcall *(x.den, y))
end
function Base.:/(x::TracedRNumber, y::TracedRational)
    return TracedRational(@opcall *(x, y.den), y.num)
end

function Base.inv(x::TracedRational)
    return TracedRational(x.den, x.num)
end

# Comparison operations
function Base.:(==)(x::TracedRational, y::TracedRational)
    return @opcall ==(x.num * y.den, y.num * x.den)
end

function Base.:<(x::TracedRational, y::TracedRational)
    # Note: assuming denominators are always positive (as they should be in a normalized Rational)
    # a/b < c/d  <==>  a*d < c*b (when b, d > 0)
    return @opcall <(x.num * y.den, y.num * x.den)
end

function Base.:<=(x::TracedRational, y::TracedRational)
    return @opcall <=(x.num * y.den, y.num * x.den)
end

# Conversion operations
function Base.float(x::TracedRational)
    return @opcall /(float(x.num), float(x.den))
end

function Base.Float64(x::TracedRational)
    return @opcall /(Float64(x.num), Float64(x.den))
end

function Base.Float32(x::TracedRational)
    return @opcall /(Float32(x.num), Float32(x.den))
end

# Type conversion
Base.convert(::Type{TracedRational{T}}, x::TracedRational{T}) where {T} = x
function Base.convert(::Type{TracedRational{T}}, x::TracedRational) where {T}
    return TracedRational{T}(x)
end
function Base.convert(::Type{TracedRational{T}}, x::Rational) where {T}
    return TracedRational{T}(x)
end
function Base.convert(::Type{TracedRational{T}}, x::Integer) where {T}
    return TracedRational{T}(x)
end

# Other utility functions
Base.zero(::Type{TracedRational{T}}) where {T} = TracedRational(zero(T), one(T))
Base.one(::Type{TracedRational{T}}) where {T} = TracedRational(one(T), one(T))
Base.zero(x::TracedRational) = zero(typeof(x))
Base.one(x::TracedRational) = one(typeof(x))

function Base.abs(x::TracedRational)
    return TracedRational(@opcall abs(x.num), @opcall abs(x.den))
end

function Base.sign(x::TracedRational)
    return @opcall sign(x.num)
end

# Rational construction with // operator
function Base.://(num::TracedRNumber, den::TracedRNumber)
    return TracedRational(num, den)
end
function Base.://(num::TracedRNumber{T}, den::Integer) where {T}
    return TracedRational(num, Reactant.promote_to(TracedRNumber{T}, den))
end
function Base.://(num::Integer, den::TracedRNumber{T}) where {T}
    return TracedRational(Reactant.promote_to(TracedRNumber{T}, num), den)
end
function Base.://(num::AbstractConcreteNumber, den::AbstractConcreteNumber)
    return TracedRational(num, den)
end

end
