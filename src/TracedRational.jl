module TracedRationalOverrides

using ..Reactant: Reactant, AbstractConcreteNumber, TracedRNumber
import ..Reactant: TracedRational
using ..Ops: @opcall
using ReactantCore: ReactantCore

ReactantCore.is_traced(::TracedRational, seen) = true
ReactantCore.is_traced(::TracedRational) = true

# Constructors
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

# Arithmetic operations
# function Base.:+(x::TracedRational, y::TracedRational)
#     num = @opcall +(x.num * y.den, y.num * x.den)
#     den = @opcall *(x.den, y.den)
#     return TracedRational(num, den)
# end

# function Base.:-(x::TracedRational, y::TracedRational)
#     num = @opcall -(x.num * y.den, y.num * x.den)
#     den = @opcall *(x.den, y.den)
#     return TracedRational(num, den)
# end

# function Base.:*(x::TracedRational, y::TracedRational)
#     num = @opcall *(x.num, y.num)
#     den = @opcall *(x.den, y.den)
#     return TracedRational(num, den)
# end

# function Base.:/(x::TracedRational, y::TracedRational)
#     num = @opcall *(x.num, y.den)
#     den = @opcall *(x.den, y.num)
#     return TracedRational(num, den)
# end

# function Base.:-(x::TracedRational)
#     return TracedRational(@opcall -(x.num), x.den)
# end

# function Base.:+(x::TracedRational, y::Integer)
#     return TracedRational(@opcall +(x.num, x.den * y), x.den)
# end
# Base.:+(y::Integer, x::TracedRational) = x + y

# function Base.:+(x::TracedRational, y::TracedRNumber)
#     return TracedRational(@opcall +(x.num, x.den * y), x.den)
# end
# Base.:+(y::TracedRNumber, x::TracedRational) = x + y

# function Base.:-(x::TracedRational, y::Integer)
#     return TracedRational(@opcall -(x.num, x.den * y), x.den)
# end
# function Base.:-(y::Integer, x::TracedRational)
#     return TracedRational(@opcall -(x.den * y, x.num), x.den)
# end

# function Base.:-(x::TracedRational, y::TracedRNumber)
#     return TracedRational(@opcall -(x.num, x.den * y), x.den)
# end
# function Base.:-(y::TracedRNumber, x::TracedRational)
#     return TracedRational(@opcall -(x.den * y, x.num), x.den)
# end

# function Base.:*(x::TracedRational, y::Integer)
#     return TracedRational(@opcall *(x.num, y), x.den)
# end
# Base.:*(y::Integer, x::TracedRational) = x * y

# function Base.:*(x::TracedRational, y::TracedRNumber)
#     return TracedRational(@opcall *(x.num, y), x.den)
# end
# Base.:*(y::TracedRNumber, x::TracedRational) = x * y

# function Base.:/(x::TracedRational, y::Integer)
#     return TracedRational(x.num, @opcall *(x.den, y))
# end
# function Base.:/(x::Integer, y::TracedRational)
#     return TracedRational(@opcall *(x, y.den), y.num)
# end

# function Base.:/(x::TracedRational, y::TracedRNumber)
#     return TracedRational(x.num, @opcall *(x.den, y))
# end
# function Base.:/(x::TracedRNumber, y::TracedRational)
#     return TracedRational(@opcall *(x, y.den), y.num)
# end

Base.inv(x::TracedRational) = TracedRational(x.den, x.num)

# # Comparison operations
# __compare(op::OP, x::TracedRational, y::TracedRational) = op(x.num * y.den, y.num * x.den)

# Base.:(==)(x::TracedRational, y::TracedRational) = __compare(==, x, y)
# Base.:<(x::TracedRational, y::TracedRational) = __compare(<, x, y)
# Base.:<=(x::TracedRational, y::TracedRational) = __compare(<=, x, y)

# Conversion operations
Base.float(x::TracedRational) = float(x.num) / float(x.den)

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

Base.abs(x::TracedRational) = TracedRational(abs(x.num), x.den)

Base.sign(x::TracedRational) = sign(x.num)

# Rational construction with // operator
function Base.://(num::TracedRNumber{<:Integer}, den::TracedRNumber{<:Integer})
    return TracedRational(num, den)
end
function Base.://(num::TracedRNumber{T}, den::Integer) where {T<:Integer}
    return TracedRational(num, Reactant.promote_to(TracedRNumber{T}, den))
end
function Base.://(num::Integer, den::TracedRNumber{T}) where {T<:Integer}
    return TracedRational(Reactant.promote_to(TracedRNumber{T}, num), den)
end
function Base.://(
    num::AbstractConcreteNumber{<:Integer}, den::AbstractConcreteNumber{<:Integer}
)
    return TracedRational(num, den)
end

end
