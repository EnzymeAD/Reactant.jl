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

# Bessel Functions of the First Kind

function SpecialFunctions.besselj0(z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_besselj(@opcall(constant(TF(0))), zf))
end

function SpecialFunctions.besselj1(z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_besselj(@opcall(constant(TF(1))), zf))
end

function SpecialFunctions.besselj(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besselj(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.besseljx(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besseljx(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.sphericalbesselj(
    nu, z::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall(special_sphericalbesselj(@opcall(constant(float(nu))), float(z)))
end

# Bessel Functions of the Second Kind

function SpecialFunctions.bessely0(z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_bessely(@opcall(constant(TF(0))), zf))
end

function SpecialFunctions.bessely1(z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_bessely(@opcall(constant(TF(1))), zf))
end

function SpecialFunctions.bessely(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_bessely(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.besselyx(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besselyx(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.sphericalbessely(
    nu, z::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall(special_sphericalbessely(@opcall(constant(float(nu))), float(z)))
end

# Bessel Functions of the Third Kind (Hankel Functions)

function SpecialFunctions.besselh(
    nu, k::Integer, z::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    nuf = @opcall(constant(float(nu)))
    zf = float(z)
    TF = eltype(zf)
    if k == 1 || k == 2
        return @opcall(special_besselh(nuf, @opcall(constant(TF(k))), zf))
    else
        throw(DomainError(k, "besselh: k must be 1 or 2"))
    end
end

function SpecialFunctions.besselh(
    nu, k::TracedRNumber{T}, z::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall(special_besselh(@opcall(constant(float(nu))), float(k), float(z)))
end

function SpecialFunctions.hankelh1(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    nuf = @opcall(constant(float(nu)))
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_besselh(nuf, @opcall(constant(TF(1))), zf))
end

function SpecialFunctions.hankelh1x(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_hankelh1x(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.hankelh2(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    nuf = @opcall(constant(float(nu)))
    zf = float(z)
    TF = eltype(zf)
    return @opcall(special_besselh(nuf, @opcall(constant(TF(2))), zf))
end

function SpecialFunctions.hankelh2x(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_hankelh2x(@opcall(constant(float(nu))), float(z)))
end

# Modified Bessel Functions of the First Kind

function SpecialFunctions.besseli(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besseli(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.besselix(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besselix(@opcall(constant(float(nu))), float(z)))
end

# Modified Bessel Functions of the Second Kind

function SpecialFunctions.besselk(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besselk(@opcall(constant(float(nu))), float(z)))
end

function SpecialFunctions.besselkx(nu, z::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return @opcall(special_besselkx(@opcall(constant(float(nu))), float(z)))
end

# Jinc Function (sombrero/besinc)

function SpecialFunctions.jinc(x::TracedRNumber{<:ReactantFloatInt})
    return @opcall(special_jinc(float(x)))
end

#Elliptic Integrals

function SpecialFunctions.zeta(
    z::TracedRNumber{T}, s::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall zeta(z, s)
end

end # module ReactantSpecialFunctionsExt
