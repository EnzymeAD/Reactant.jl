module ReactantSpecialFunctionsExt
using SpecialFunctions
using Reactant: Ops, Reactant, TracedRNumber, ReactantFloat
using Reactant.TracedRNumberOverrides: float

for fn in [:digamma, :erf, :erfc, (:loggamma, :lgamma)]
    (fns, fno) = fn isa Tuple ? fn : (fn, fn)
    @eval(function SpecialFunctions.$fns(x::TracedRNumber{<:Real})
        return Ops.$fno(float(x))
    end)
end

function SpecialFunctions.gamma(x::TracedRNumber{<:Real})
    return exp(Ops.lgamma(float(x)))
end

function SpecialFunctions.gamma(n::TracedRNumber{<:Integer})
    return round(gamma(float(n)))
end

function SpecialFunctions.loggamma1p(x::TracedRNumber{<:Real})
    return loggamma(1 + x)
end

function SpecialFunctions.logfactorial(x::TracedRNumber{<:Integer})
    return loggamma(1 + x)
end

# SpecialFunctions.invdigamma

function SpecialFunctions.trigamma(x::TracedRNumber{<:Real})
    return Ops.polygamma(Ops.constant(Float64(1)), float(x))#TODO: change Ops definition
end

function SpecialFunctions.polygamma(n::TracedRNumber{<:Real}, x::TracedRNumber{<:Real})
    return Ops.polygamma(float(n), float(x))
end

# SpecialFunctions.gamma_inc

# SpecialFunctions.gamma_inc_inv

function SpecialFunctions.loggammadiv(
    a::TracedRNumber{T}, b::TracedRNumber{T}
) where {T<:Real}
    return log(gamma(b) / gamma(a + b))
end

#SpecialFunctions.gamma ...

function SpecialFunctions.beta(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return gamma(x) * gamma(y) / gamma(x + y)
end

function SpecialFunctions.logbeta(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return log(abs(beta(x, y)))
end

#TODO: sign function
#SpecialFunctions.logabsbeta
#SpecialFunctions.logabsbinomial

#SpecialFunctions.beta...

#utilities...

function SpecialFunctions.erf(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return erf(y) - erf(x)
end

#SpecialFunctions.erfcinv

function SpecialFunctions.logerf(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return log(erf(x, y))
end

function SpecialFunctions.erfcx(x::TracedRNumber{<:Real})
    return exp(float(x^2)) * erfc(x)
end

function SpecialFunctions.logerfc(x::TracedRNumber{<:Real})
    return log(erfc(x))
end

function SpecialFunctions.logerfcx(x::TracedRNumber{<:Real})
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

function SpecialFunctions.zeta(z::TracedRNumber{T}, s::TracedRNumber{T}) where {T<:Real}
    return Ops.zeta(z, s)
end

end # module ReactantSpecialFunctionsExt