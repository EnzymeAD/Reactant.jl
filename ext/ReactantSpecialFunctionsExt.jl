module ReactantSpecialFunctionsExt

using SpecialFunctions: SpecialFunctions
using Reactant: Ops, Reactant, TracedRNumber, ReactantFloat, ReactantInt, ReactantFloatInt
using Reactant.Ops: @opcall

for fn in [:digamma, :erf, :erfc, (:loggamma, :lgamma)]
    (fns, fno) = fn isa Tuple ? fn : (fn, fn)
    @eval(function SpecialFunctions.$fns(x::TracedRNumber{<:ReactantFloatInt})
        return @opcall $fno(float(x))
    end)
end

function SpecialFunctions.gamma(x::TracedRNumber{<:ReactantFloat})
    return exp(@opcall(lgamma(float(x))))
end

function SpecialFunctions.gamma(n::TracedRNumber{<:ReactantInt})
    return round(SpecialFunctions.gamma(float(n)))
end

function SpecialFunctions.loggamma1p(x::TracedRNumber{<:ReactantFloat})
    return SpecialFunctions.loggamma(one(x) + x)
end

function SpecialFunctions.logfactorial(x::TracedRNumber{<:ReactantInt})
    return SpecialFunctions.loggamma(one(x) + x)
end

# SpecialFunctions.invdigamma

function SpecialFunctions.trigamma(x::TracedRNumber{<:ReactantFloatInt})
    #TODO: change Ops definition
    return @opcall(polygamma(@opcall(constant(Float64(1))), float(x)))
end

function SpecialFunctions.polygamma(
    n::TracedRNumber{<:ReactantFloatInt}, x::TracedRNumber{<:ReactantFloatInt}
)
    return @opcall polygamma(float(n), float(x))
end

# SpecialFunctions.gamma_inc

# SpecialFunctions.gamma_inc_inv

function SpecialFunctions.loggammadiv(
    a::TracedRNumber{T}, b::TracedRNumber{T}
) where {T<:ReactantFloat}
    return log(SpecialFunctions.gamma(b) / SpecialFunctions.gamma(a + b))
end

#SpecialFunctions.gamma ...

function SpecialFunctions.beta(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return SpecialFunctions.gamma(x) * SpecialFunctions.gamma(y) /
           SpecialFunctions.gamma(x + y)
end

function SpecialFunctions.logbeta(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return log(abs(SpecialFunctions.beta(x, y)))
end

#TODO: sign function
#SpecialFunctions.logabsbeta
#SpecialFunctions.logabsbinomial

#SpecialFunctions.beta...

#utilities...

function SpecialFunctions.erf(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return SpecialFunctions.erf(y) - SpecialFunctions.erf(x)
end

#SpecialFunctions.erfcinv

function SpecialFunctions.logerf(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return log(SpecialFunctions.erf(x, y))
end

function SpecialFunctions.erfcx(x::TracedRNumber{<:ReactantFloatInt})
    return exp(float(x^2)) * SpecialFunctions.erfc(x)
end

function SpecialFunctions.logerfc(x::TracedRNumber{<:ReactantFloatInt})
    return log(SpecialFunctions.erfc(x))
end

function SpecialFunctions.logerfcx(x::TracedRNumber{<:ReactantFloatInt})
    return log(SpecialFunctions.erfcx(x))
end

#Unsupported complex
#SpecialFunctions.erfi

#SpecialFunctions.erfinv
#SpecialFunctions.dawson
#SpecialFunctions.faddeeva

#Airy and Related Functions

#Bessel ...

#Elliptic Integrals

function SpecialFunctions.zeta(
    z::TracedRNumber{T}, s::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall zeta(z, s)
end

end # module ReactantSpecialFunctionsExt
