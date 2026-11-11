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
    #TODO(#2262): change Ops definition
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

#TODO(#2262): sign function
#SpecialFunctions.logabsbeta
#SpecialFunctions.logabsbinomial

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

const _WEIDEMAN_L = 5.3182958969449885
const _INV_SQRT_PI = 0.5641895835477563
const _TWO_INV_SQRT_PI = 1.1283791670955126
const _HALF_SQRT_PI = 0.886226925452758

const _WEIDEMAN_A = (
    2.8996245093897053,
    2.61605415276186,
    2.201513794878312,
    1.7253830848179779,
    1.2563815675765133,
    0.8472174576593818,
    0.5266528988277086,
    0.29989437996150065,
    0.15504263802479495,
    0.07182361779074337,
    0.029202916471241867,
    0.010048186242783424,
    0.0027054056330737914,
    0.0004398070159869668,
    -3.939363145489569e-5,
    -5.591309264248318e-5,
    -1.8007447144750956e-5,
    -1.0660138984947143e-6,
    1.483566113220078e-6,
    5.912136951899494e-7,
    1.4198642399935674e-8,
    -6.35177348504429e-8,
    -1.8315616783040462e-8,
    3.2497465180436973e-9,
    3.0177805400090707e-9,
    2.1086006347066517e-10,
    -3.5632339865976533e-10,
    -9.055124450928292e-11,
    3.47272670930455e-11,
    1.7714495214011192e-11,
    -2.7276023158200452e-12,
    -2.907688342182867e-12,
    1.2031458219387989e-13,
    4.5329666782606727e-13,
    1.37256205867155e-14,
    -7.074086260286855e-14,
    -5.409310282882142e-15,
    1.1357687198999241e-14,
    1.128073562364402e-15,
    -1.899694947394927e-15,
)

const _ERF_MACLAURIN_C = (
    1.0,
    -0.3333333333333333,
    0.1,
    -0.023809523809523808,
    0.004629629629629629,
    -0.0007575757575757576,
    0.00010683760683760684,
    -1.3227513227513228e-5,
    1.4589169000933706e-6,
    -1.4503852223150468e-7,
    1.3122532963802806e-8,
    -1.0892221037148573e-9,
    8.35070279514724e-11,
    -5.9477940136376354e-12,
    3.9554295164585257e-13,
    -2.466827010264457e-14,
    1.4483264643598138e-15,
)

const _DAWSON_MACLAURIN_C = (
    1.0,
    -0.6666666666666666,
    0.26666666666666666,
    -0.0761904761904762,
    0.016931216931216932,
    -0.0030784030784030783,
    0.0004736004736004736,
    -6.314672981339648e-5,
    7.4290270368701745e-6,
    -7.820028459863341e-7,
    7.447646152250801e-8,
    -6.476214045435479e-9,
    5.180971236348383e-10,
    -3.8377564713691727e-11,
    2.6467286009442573e-12,
)

_complex_erf_work_type(::Type{Float64}) = Float64
_complex_erf_work_type(::Type{T}) where {T<:Integer} = Float64
_complex_erf_work_type(::Type{T}) where {T} = Float32

function _faddeeva_uhp(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    CT = Complex{T}
    L_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(_WEIDEMAN_L), zero(T)))
    iz = complex(-imag(z), real(z))
    Lmiz = L_c - iz
    Lpiz = L_c + iz
    t = Lpiz / Lmiz
    p = Reactant.promote_to(TracedRNumber{CT}, CT(T(_WEIDEMAN_A[end]), zero(T)))
    for k in (length(_WEIDEMAN_A) - 1):-1:1
        ck = Reactant.promote_to(TracedRNumber{CT}, CT(T(_WEIDEMAN_A[k]), zero(T)))
        p = p * t + ck
    end
    two_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(2), zero(T)))
    inv_sqrt_pi_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(_INV_SQRT_PI), zero(T)))
    return (two_c / (Lmiz * Lmiz)) * p + inv_sqrt_pi_c / Lmiz
end

function _faddeeva_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    CT = Complex{T}
    zero_z = Reactant.promote_to(TracedRNumber{CT}, zero(CT))
    neg_im = imag(z) < zero(T)
    z_uhp = ifelse(neg_im, -z, z)
    w_pos = _faddeeva_uhp(z_uhp)
    z_safe = ifelse(neg_im, z, zero_z)
    two_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(2), zero(T)))
    return ifelse(neg_im, two_c * exp(-(z_safe * z_safe)) - w_pos, w_pos)
end

function _erfcx_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    return _faddeeva_complex(complex(-imag(z), real(z)))
end

function _erfc_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    CT = Complex{T}
    neg_re = real(z) < zero(T)
    z_rhp = ifelse(neg_re, -z, z)
    c_rhp = exp(-(z_rhp * z_rhp)) * _erfcx_complex(z_rhp)
    two_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(2), zero(T)))
    return ifelse(neg_re, two_c - c_rhp, c_rhp)
end

function _erf_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    CT = Complex{T}
    zero_z = Reactant.promote_to(TracedRNumber{CT}, zero(CT))
    one_c = Reactant.promote_to(TracedRNumber{CT}, one(CT))
    small = (real(z) * real(z) + imag(z) * imag(z)) < T(0.25)
    z_small = ifelse(small, z, zero_z)
    z2 = z_small * z_small
    s = Reactant.promote_to(TracedRNumber{CT}, CT(T(_ERF_MACLAURIN_C[end]), zero(T)))
    for k in (length(_ERF_MACLAURIN_C) - 1):-1:1
        ck = Reactant.promote_to(TracedRNumber{CT}, CT(T(_ERF_MACLAURIN_C[k]), zero(T)))
        s = s * z2 + ck
    end
    scale_c = Reactant.promote_to(TracedRNumber{CT}, CT(T(_TWO_INV_SQRT_PI), zero(T)))
    erf_small = (scale_c * z_small) * s

    neg_re = real(z) < zero(T)
    z_rhp = ifelse(neg_re, -z, z)
    erf_large_rhp = one_c - exp(-(z_rhp * z_rhp)) * _erfcx_complex(z_rhp)
    erf_large = ifelse(neg_re, -erf_large_rhp, erf_large_rhp)
    return ifelse(small, erf_small, erf_large)
end

function _erfi_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    w = _erf_complex(complex(-imag(z), real(z)))
    return complex(imag(w), -real(w))
end

function _dawson_complex(z::TracedRNumber{Complex{T}}) where {T<:Union{Float32,Float64}}
    CT = Complex{T}
    zero_z = Reactant.promote_to(TracedRNumber{CT}, zero(CT))
    small = (real(z) * real(z) + imag(z) * imag(z)) < T(0.04)
    z_small = ifelse(small, z, zero_z)
    z2 = z_small * z_small
    p = Reactant.promote_to(TracedRNumber{CT}, CT(T(_DAWSON_MACLAURIN_C[end]), zero(T)))
    for k in (length(_DAWSON_MACLAURIN_C) - 1):-1:1
        ck = Reactant.promote_to(TracedRNumber{CT}, CT(T(_DAWSON_MACLAURIN_C[k]), zero(T)))
        p = p * z2 + ck
    end
    daw_small = z_small * p

    neg_im = imag(z) < zero(T)
    z_uhp = ifelse(neg_im, -z, z)
    diff_w = _faddeeva_complex(z_uhp) - exp(-(z_uhp * z_uhp))
    hsp = T(_HALF_SQRT_PI)
    daw_uhp = complex(imag(diff_w) * hsp, -real(diff_w) * hsp)
    daw_large = ifelse(neg_im, -daw_uhp, daw_uhp)
    return ifelse(small, daw_small, daw_large)
end

for (fn, impl) in (
    (:faddeeva, :_faddeeva_complex),
    (:erfcx, :_erfcx_complex),
    (:erfc, :_erfc_complex),
    (:erf, :_erf_complex),
    (:erfi, :_erfi_complex),
    (:dawson, :_dawson_complex),
)
    @eval function SpecialFunctions.$fn(
        z::TracedRNumber{Complex{T}}
    ) where {T<:ReactantFloatInt}
        WT = _complex_erf_work_type(T)
        FT = float(T)
        zw = Reactant.promote_to(TracedRNumber{Complex{WT}}, z)
        res = $impl(zw)
        return WT === FT ? res : Reactant.promote_to(TracedRNumber{Complex{FT}}, res)
    end
end

function SpecialFunctions.erf(
    x::TracedRNumber{T1}, y::TracedRNumber{T2}
) where {T1<:Complex,T2<:Union{ReactantFloatInt,Complex}}
    return SpecialFunctions.erf(y) - SpecialFunctions.erf(x)
end
function SpecialFunctions.erf(
    x::TracedRNumber{T1}, y::TracedRNumber{T2}
) where {T1<:ReactantFloatInt,T2<:Complex}
    return SpecialFunctions.erf(y) - SpecialFunctions.erf(x)
end

SpecialFunctions.logerfc(z::TracedRNumber{<:Complex}) = log(SpecialFunctions.erfc(z))
SpecialFunctions.logerfcx(z::TracedRNumber{<:Complex}) = log(SpecialFunctions.erfcx(z))
function SpecialFunctions.logerf(
    x::TracedRNumber{T1}, y::TracedRNumber{T2}
) where {T1<:Complex,T2<:Union{ReactantFloatInt,Complex}}
    return log(SpecialFunctions.erf(x, y))
end
function SpecialFunctions.logerf(
    x::TracedRNumber{T1}, y::TracedRNumber{T2}
) where {T1<:ReactantFloatInt,T2<:Complex}
    return log(SpecialFunctions.erf(x, y))
end

function SpecialFunctions.erfi(x::TracedRNumber{<:ReactantFloatInt})
    xf = float(x)
    zf = complex(xf, zero(xf))
    return real(SpecialFunctions.erfi(zf))
end

function SpecialFunctions.dawson(x::TracedRNumber{<:ReactantFloatInt})
    xf = float(x)
    zf = complex(xf, zero(xf))
    return real(SpecialFunctions.dawson(zf))
end

function SpecialFunctions.faddeeva(x::TracedRNumber{<:ReactantFloatInt})
    xf = float(x)
    return SpecialFunctions.faddeeva(complex(xf, zero(xf)))
end

#Airy and Related Functions

#Bessel ...

#Elliptic Integrals

function SpecialFunctions.zeta(
    z::TracedRNumber{T}, s::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall zeta(z, s)
end

end # module ReactantSpecialFunctionsExt
