using Reactant: Reactant, Ops
using Reactant.TracedUtils

function SpecialFunctions.gamma(x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return exp(Ops.lgamma(x))
end

#TODO: add factorial function
#=function SpecialFunctions.gamma(
    n::TracedRNumber{T}
) where {T<:Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}}
     factorial(n)
end
=#

function SpecialFunctions.loggamma(x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return Ops.lgamma(x)
end

function SpecialFunctions.loggamma1p(x::TracedRNumber{T}) where {T}
    return loggamma(1 + x)
end

function SpecialFunctions.logfactorial(
    x::TracedRNumber{<:T}
) where {T<:Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}}
    return loggamma(1 + x)
end

function SpecialFunctions.digamma(x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return Ops.digamma(x)
end

# SpecialFunctions.invdigamma

function SpecialFunctions.trigamma(x::TracedRNumber{T}) where {T}
    return Ops.polygamma(Ops.constant(2.0), x)
end

function SpecialFunctions.polygamma(n::TracedRNumber{T}, x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return Ops.polygamma(n, x)
end

# SpecialFunctions.gamma_inc

# SpecialFunctions.gamma_inc_inv

function SpecialFunctions.loggammadiv(a::TracedRNumber{T}, b::TracedRNumber{T}) where {T}
    return log(gamma(b) / gamma(a + b))
end

#SpecialFunctions.gamma ...

function SpecialFunctions.beta(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T}
    return gamma(x) * gamma(y) / gamma(x + y)
end

function SpecialFunctions.logbeta(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T}
    return log(abs(beta(x, y)))
end

#TODO: sign function
#SpecialFunctions.logabsbeta
#SpecialFunctions.logabsbinomial

#SpecialFunctions.beta...

#utilities...

function SpecialFunctions.erf(x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return Ops.erf(x)
end

function SpecialFunctions.erf(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T}
    return y - x
end

function SpecialFunctions.erfc(x::TracedRNumber{T}) where {T}
    x = promote_to(TracedRNumber{Float64}, x)
    return Ops.erfc(x)
end

function SpecialFunctions.logerf(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T}
    return log(erf(x, y))
end

#SpecialFunctions.erfcinv

function SpecialFunctions.logerf(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T}
    return log(erf(x, y))
end

function SpecialFunctions.erfcx(x::TracedRNumber{T}) where {T}
    return exp(x^2) * erfc(x)
end

function SpecialFunctions.logerfc(x::TracedRNumber{T}) where {T}
    return log(erfc(x))
end

function SpecialFunctions.logerfcx(x::TracedRNumber{T}) where {T}
    return log(erfcx(x))
end

#Unsupported complex
#SpecialFunctions.erfi

#SpecialFunctions.erfinv
#SpecialFunctions.dawson
#SpecialFunctions.faddeeva

#Airy and Related Functions

#Bessel ...

#Elliptic Integrals

function SpecialFunctions.zeta(z::TracedRNumber{T}, s::TracedRNumber{T}) where {T}
    return Ops.zeta(z,s)
end