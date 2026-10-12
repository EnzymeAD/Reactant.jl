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

# ---------------------------------------------------------------------------
# Complex error functions
#
# `chlo.erf` / `chlo.erfc` only accept floating point operands. For complex
# arguments we instead evaluate the Faddeeva function
#     w(z) = exp(-z^2) * erfc(-im * z)
# with Weideman's rational approximation (J.A.C. Weideman, "Computation of the
# complex error function", SIAM J. Numer. Anal. 31 (1994), pp. 1497-1518) and
# derive erf / erfc / erfcx / erfi from it. The approximation is a fixed size
# polynomial in Z = (L + im z) / (L - im z) so the whole evaluation is
# branch-free: every region reduction (half-plane symmetries, Taylor series near
# the origin, overflow guard) is expressed with `ifelse`, which makes it
# traceable and differentiable. The inputs of non-selected branches are clamped
# to benign values so that neither the primal nor its (Enzyme) derivatives
# produce Inf/NaN from the branch that is discarded.
#
# Accuracy (relative to max(|f|, 1) on re, im ∈ [-12, 12]): ~1e-14 for
# ComplexF64 (N = 48) and ~3e-7 for ComplexF32 (N = 20).
# ---------------------------------------------------------------------------

# Coefficients `a_n` and scale `L` of Weideman's approximation
#     w(z) ≈ 1 / (√π (L - im z)) + 2 / (L - im z)^2 * Σ_{n=1}^{N} a_n Z^{n-1}
function weideman_coefficients(::Type{T}, N::Int) where {T<:AbstractFloat}
    M = 2N
    L = sqrt(N / sqrt(2.0))
    f(k) = begin
        t = L * tan(k * pi / (2M))
        return exp(-t^2) * (L^2 + t^2)
    end
    a = Vector{T}(undef, N)
    for n in 1:N
        s = f(0)
        for k in 1:(M - 1)
            s += 2 * f(k) * cos(k * n * pi / M)
        end
        a[n] = T(s / (2M))
    end
    return T(L), a
end

# Taylor coefficients of erf(z) = Σ_n c_n z^(2n+1)
function erf_taylor_coefficients(::Type{T}, nterms::Int) where {T<:AbstractFloat}
    return T[
        T(2 / sqrt(big(pi)) * (-1)^n / (factorial(big(n)) * (2n + 1))) for
        n in 0:(nterms - 1)
    ]
end

const WEIDEMAN_F64 = weideman_coefficients(Float64, 48)
const WEIDEMAN_F32 = weideman_coefficients(Float32, 20)
const ERF_TAYLOR_F64 = erf_taylor_coefficients(Float64, 20)
const ERF_TAYLOR_F32 = erf_taylor_coefficients(Float32, 11)

weideman_coefficients(::Type{Float64}) = WEIDEMAN_F64
weideman_coefficients(::Type{Float32}) = WEIDEMAN_F32
erf_taylor_coefficients(::Type{Float64}) = ERF_TAYLOR_F64
erf_taylor_coefficients(::Type{Float32}) = ERF_TAYLOR_F32

# Below this value of real(-z^2) `exp(-z^2)` underflows to zero (and the
# intermediate values in the rational approximation may overflow)
exp_underflow_threshold(::Type{T}) where {T} = -T(1.05) * log(floatmax(T))

real_type(::TracedRNumber{Complex{T}}) where {T} = T
real_type(::Complex{T}) where {T} = T

mul_im(z) = Complex(-imag(z), real(z)) # im * z
mul_neg_im(z) = Complex(imag(z), -real(z)) # -im * z
square(z) = z * z

# w(z) for imag(z) >= 0
function faddeeva_upper_half_plane(z)
    T = real_type(z)
    L, a = weideman_coefficients(T)
    iz = mul_im(z)
    d = inv(L - iz)
    Z = (L + iz) * d
    p = a[end] * one(z)
    for n in (lastindex(a) - 1):-1:firstindex(a)
        p = p * Z + a[n]
    end
    return (2 * p * d + T(1 / sqrt(pi))) * d
end

# erf(z) via its Taylor series around 0 (accurate for |z| <= 1)
function erf_taylor(z)
    T = real_type(z)
    c = erf_taylor_coefficients(T)
    z2 = square(z)
    s = c[end] * one(z)
    for n in (lastindex(c) - 1):-1:firstindex(c)
        s = s * z2 + c[n]
    end
    return z * s
end

function complex_faddeeva(z)
    upper = imag(z) >= 0
    zp = ifelse(upper, z, -z)
    w = faddeeva_upper_half_plane(zp)
    # w(-z) = 2 exp(-z^2) - w(z); clamp the discarded branch to keep it finite
    zl = ifelse(upper, zero(z), z)
    return ifelse(upper, w, 2 * exp(-square(zl)) - w)
end

# erfc(z) = exp(-z^2) w(im z) for real(z) >= 0; erfc(-z) = 2 - erfc(z)
# Returns `(right, e)` where `right = real(z) >= 0` and `e = erfc(±z)` for the
# reflected argument with non-negative real part.
function complex_erfc_right_half_plane(z)
    T = real_type(z)
    right = real(z) >= 0
    zp = ifelse(right, z, -z)
    mz2 = -square(zp)
    # Guard against overflow/NaN in the rational approximation for huge |z|
    # (where erfc is 0 anyways). Clamp the input so that the derivative of the
    # discarded branch stays finite.
    tiny = real(mz2) < exp_underflow_threshold(T)
    zs = ifelse(tiny, one(zp), zp)
    e = exp(-square(zs)) * faddeeva_upper_half_plane(mul_im(zs))
    return right, ifelse(tiny, zero(e), e)
end

function complex_erf(z)
    right, e = complex_erfc_right_half_plane(z)
    # 1 - e loses relative precision near the origin, use the Taylor series
    small = (square(real(z)) + square(imag(z))) < 1
    zt = ifelse(small, z, zero(z))
    return ifelse(small, erf_taylor(zt), ifelse(right, one(e) - e, e - one(e)))
end

function complex_erfc(z)
    right, e = complex_erfc_right_half_plane(z)
    return ifelse(right, e, 2 - e)
end

for fn in (:erf, :erfc)
    cfn = Symbol(:complex_, fn)
    @eval function SpecialFunctions.$fn(z::TracedRNumber{<:Reactant.ReactantComplexFloat})
        return $cfn(z)
    end
end

function SpecialFunctions.faddeeva(z::TracedRNumber{<:Reactant.ReactantComplexFloat})
    return complex_faddeeva(z)
end

function SpecialFunctions.faddeeva(x::TracedRNumber{<:ReactantFloatInt})
    return SpecialFunctions.faddeeva(Complex(float(x)))
end

# erfcx(z) = exp(z^2) erfc(z) = w(im z)
function SpecialFunctions.erfcx(z::TracedRNumber{<:Reactant.ReactantComplexFloat})
    return complex_faddeeva(mul_im(z))
end

# erfi(z) = -im erf(im z)
function SpecialFunctions.erfi(z::TracedRNumber{<:Reactant.ReactantComplexFloat})
    return mul_neg_im(complex_erf(mul_im(z)))
end

function SpecialFunctions.erfi(x::TracedRNumber{<:ReactantFloatInt})
    return real(SpecialFunctions.erfi(Complex(float(x))))
end

#SpecialFunctions.dawson

#Airy and Related Functions

#Bessel ...

#Elliptic Integrals

function SpecialFunctions.zeta(
    z::TracedRNumber{T}, s::TracedRNumber{T}
) where {T<:ReactantFloatInt}
    return @opcall zeta(z, s)
end

end # module ReactantSpecialFunctionsExt
