module ReactantSpecialFunctionsExt

using SpecialFunctions: SpecialFunctions
using Reactant: Ops, Reactant, TracedRNumber, ReactantFloat, ReactantInt, ReactantFloatInt
using Reactant.Ops: @opcall

for fn in [:digamma, :erf, :erfc, (:loggamma, :lgamma), (:erfinv, :erf_inv)]
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
    return @opcall polygamma(1, float(x))
end

function SpecialFunctions.polygamma(
    n::TracedRNumber{<:ReactantInt}, x::TracedRNumber{<:ReactantFloatInt}
)
    return @opcall polygamma(n, float(x))
end

function SpecialFunctions.polygamma(n::Integer, x::TracedRNumber{<:ReactantFloatInt})
    return @opcall polygamma(n, float(x))
end

function SpecialFunctions.polygamma(
    n::TracedRNumber{<:ReactantFloat}, x::TracedRNumber{<:ReactantFloatInt}
)
    fx = float(x)
    T = promote_type(Reactant.unwrapped_eltype(n), Reactant.unwrapped_eltype(fx))
    return @opcall polygamma(
        Reactant.promote_to(TracedRNumber{T}, n), Reactant.promote_to(TracedRNumber{T}, fx)
    )
end

# SpecialFunctions.gamma_inc

# SpecialFunctions.gamma_inc_inv

function SpecialFunctions.loggammadiv(
    a::TracedRNumber{T}, b::TracedRNumber{T}
) where {T<:ReactantFloat}
    return @opcall(lgamma(b)) - @opcall(lgamma(a + b))
end

function SpecialFunctions.logabsgamma(x::TracedRNumber{<:ReactantFloatInt})
    fx = float(x)
    raw_lg = @opcall lgamma(fx)
    lg = ifelse(isinf(fx), oftype(fx, Inf), raw_lg)
    fl = floor(fx)
    neg_nonint = (fx < zero(fx)) & isfinite(fx) & (fx != fl)
    odd_fl = rem(abs(fl), oftype(fx, 2)) == one(fx)
    pos_one = Reactant.promote_to(TracedRNumber{Int}, 1)
    neg_one = Reactant.promote_to(TracedRNumber{Int}, -1)
    sgn = ifelse(neg_nonint & odd_fl, neg_one, pos_one)
    return (lg, sgn)
end

function SpecialFunctions.beta(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return SpecialFunctions.gamma(x) * SpecialFunctions.gamma(y) /
           SpecialFunctions.gamma(x + y)
end

function SpecialFunctions.logbeta(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return SpecialFunctions.logabsbeta(x, y)[1]
end

function SpecialFunctions.logabsbeta(
    a::TracedRNumber{<:ReactantFloatInt}, b::TracedRNumber{<:ReactantFloatInt}
)
    FT = promote_type(
        Reactant.unwrapped_eltype(float(a)), Reactant.unwrapped_eltype(float(b))
    )
    fa = Reactant.promote_to(TracedRNumber{FT}, float(a))
    fb = Reactant.promote_to(TracedRNumber{FT}, float(b))
    lo = min(fa, fb)
    hi = max(fa, fb)

    lo_nonpos_int = (lo <= zero(lo)) & isfinite(lo) & (lo == floor(lo))
    both_pole_case =
        lo_nonpos_int & ((lo + hi) <= zero(lo)) & isfinite(hi) & (hi == floor(hi))

    # Guard arguments for the reflection branch: logbeta(1 - lo - hi, hi)
    refl_a = ifelse(both_pole_case, one(lo) - lo - hi, one(lo))
    refl_b = ifelse(both_pole_case, hi, one(hi))
    r_refl_a, _ = SpecialFunctions.logabsgamma(refl_a)
    r_refl_b, _ = SpecialFunctions.logabsgamma(refl_b)
    r_refl_ab, _ = SpecialFunctions.logabsgamma(refl_a + refl_b)
    val_refl = r_refl_a + r_refl_b - r_refl_ab

    pos_one = Reactant.promote_to(TracedRNumber{Int}, 1)
    neg_one = Reactant.promote_to(TracedRNumber{Int}, -1)
    hi_even = rem(abs(hi), oftype(hi, 2)) == zero(hi)
    sgn_refl = ifelse(hi_even, pos_one, neg_one)

    ya, sa = SpecialFunctions.logabsgamma(lo)
    yb, sb = SpecialFunctions.logabsgamma(hi)
    yab, sab = SpecialFunctions.logabsgamma(lo + hi)
    val_gen = ya + yb - yab
    sgn_gen = sa * sb * sab

    inf_val = Reactant.promote_to(TracedRNumber{FT}, FT(Inf))
    val = ifelse(lo_nonpos_int, ifelse(both_pole_case, val_refl, inf_val), val_gen)
    sgn = ifelse(lo_nonpos_int, ifelse(both_pole_case, sgn_refl, pos_one), sgn_gen)
    return (val, sgn)
end

function SpecialFunctions.logabsbeta(a::Real, b::TracedRNumber{<:ReactantFloatInt})
    FT = promote_type(typeof(float(a)), Reactant.unwrapped_eltype(float(b)))
    return SpecialFunctions.logabsbeta(
        Reactant.promote_to(TracedRNumber{FT}, FT(a)),
        Reactant.promote_to(TracedRNumber{FT}, float(b)),
    )
end

function SpecialFunctions.logabsbeta(a::TracedRNumber{<:ReactantFloatInt}, b::Real)
    FT = promote_type(Reactant.unwrapped_eltype(float(a)), typeof(float(b)))
    return SpecialFunctions.logabsbeta(
        Reactant.promote_to(TracedRNumber{FT}, float(a)),
        Reactant.promote_to(TracedRNumber{FT}, FT(b)),
    )
end

function SpecialFunctions.logabsbinomial(
    n::TracedRNumber{<:ReactantInt}, k::TracedRNumber{<:ReactantInt}
)
    IT = promote_type(Reactant.unwrapped_eltype(n), Reactant.unwrapped_eltype(k))
    nn0 = Reactant.promote_to(TracedRNumber{IT}, n)
    kk0 = Reactant.promote_to(TracedRNumber{IT}, k)
    S = float(IT)

    neg_n = nn0 < zero(nn0)
    nn = ifelse(neg_n, -nn0 + kk0 - one(nn0), nn0)
    k_odd = rem(kk0, IT(2)) != zero(kk0)
    pos_s = Reactant.promote_to(TracedRNumber{S}, one(S))
    neg_s = Reactant.promote_to(TracedRNumber{S}, -one(S))
    zero_s = Reactant.promote_to(TracedRNumber{S}, zero(S))
    neg_inf_s = Reactant.promote_to(TracedRNumber{S}, typemin(S))
    s_base = ifelse(neg_n & k_odd, neg_s, pos_s)

    is_zero_binom = (kk0 < zero(kk0)) | (kk0 > nn)
    is_unit_binom = (kk0 == zero(kk0)) | (kk0 == nn)

    kk = ifelse(kk0 > div(nn, IT(2)), nn - kk0, kk0)
    nn_safe = max(nn, IT(2))
    kk_safe = min(max(kk, one(IT)), nn_safe - one(IT))

    val_k1 = log(abs(float(nn_safe)))
    fa = float(nn_safe - kk_safe + one(IT))
    fb = float(kk_safe + one(IT))
    lb = @opcall(lgamma(fa)) + @opcall(lgamma(fb)) - @opcall(lgamma(fa + fb))
    val_gen = -log1p(float(nn_safe)) - lb

    val = ifelse(
        is_zero_binom,
        neg_inf_s,
        ifelse(is_unit_binom, zero_s, ifelse(kk == one(IT), val_k1, val_gen)),
    )
    sgn = ifelse(is_zero_binom, zero_s, s_base)
    return (val, sgn)
end

function SpecialFunctions.logabsbinomial(n::Integer, k::TracedRNumber{<:ReactantInt})
    IT = promote_type(typeof(n), Reactant.unwrapped_eltype(k))
    return SpecialFunctions.logabsbinomial(
        Reactant.promote_to(TracedRNumber{IT}, IT(n)),
        Reactant.promote_to(TracedRNumber{IT}, k),
    )
end

function SpecialFunctions.logabsbinomial(n::TracedRNumber{<:ReactantInt}, k::Integer)
    IT = promote_type(Reactant.unwrapped_eltype(n), typeof(k))
    return SpecialFunctions.logabsbinomial(
        Reactant.promote_to(TracedRNumber{IT}, n),
        Reactant.promote_to(TracedRNumber{IT}, IT(k)),
    )
end

#SpecialFunctions.beta...

#utilities...

function SpecialFunctions.erf(
    x::TracedRNumber{T}, y::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return SpecialFunctions.erf(y) - SpecialFunctions.erf(x)
end

function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:ReactantFloatInt}
    return SpecialFunctions.erfinv(one(x) - x)
end

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
