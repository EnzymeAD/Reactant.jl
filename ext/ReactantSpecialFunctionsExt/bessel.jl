# Integer-order, real-argument exponentially scaled modified Bessel I.
# For small |x| use the positive series DLMF 10.25.2, summed relative to
# its first term so the prefactor cannot underflow before summation.
# For large |x| use DLMF 10.41.3 and 10.41.9. Writing U_k(p)/n^k as
# P_k(p^2)/hypot(n,x)^k also covers n=0 without division by the order.
# The fixed coefficients below follow the polynomial recurrence 10.41.9.
const _BESSELIX_DEBYE = (
    (1 / 8, -5 / 24),
    (9 / 128, -77 / 192, 385 / 1152),
    (75 / 1024, -4563 / 5120, 17017 / 9216, -85085 / 82944),
    (3675 / 32768, -96833 / 40960, 144001 / 16384, -7436429 / 663552, 37182145 / 7962624),
    (
        59535 / 262144,
        -67608983 / 9175040,
        250881631 / 5898240,
        -108313205 / 1179648,
        5391411025 / 63700992,
        -5391411025 / 191102976,
    ),
    (
        2401245 / 4194304,
        -388895895 / 14680064,
        1441372804469 / 6606028800,
        -33010308331 / 47185920,
        4445922195 / 4194304,
        -1169936192425 / 1528823808,
        5849680962125 / 27518828544,
    ),
    (
        57972915 / 33554432,
        -25388505925 / 234881024,
        1007390378503 / 838860800,
        -1602251736839 / 301989888,
        10559432785187 / 905969664,
        -36927006432745 / 2717908992,
        1774793203908725 / 220150628352,
        -1267709431363375 / 660451885056,
    ),
    (
        13043905875 / 2147483648,
        -928090660435 / 1879048192,
        667955999804539 / 93952409600,
        -276439228010667 / 6710886400,
        3542717254441859 / 28991029248,
        -39803268297948155 / 195689447424,
        75358832548684685 / 391378894848,
        -512408152157076175 / 5283615080448,
        2562040760785380875 / 126806761930752,
    ),
)

function _besselix_series(v, a)
    T = Reactant.unwrapped_eltype(a)
    term = one(a)
    total = term * one(a)
    k = 0
    # At a <= 64 the remaining positive tail after 128 terms is well below
    # Float64 precision. Keep a runtime loop with a fixed numerical budget.
    Reactant.@trace while k < 128
        k = k + 1
        term = term * (a / 2)^2 / (oftype(a, k) * (v + k))
        total = total + term
    end
    Reactant.@trace if v == 0
        result = exp(-a) * total
    elseif v == 1
        result = (a / 2) * exp(-a) * total
    elseif v == 2
        result = ((a / 2)^2 / 2) * exp(-a) * total
    else
        logprefactor = -a + v * (log(a) - T(log(2))) - SpecialFunctions.loggamma(v + 1)
        result = exp(logprefactor + log(total))
    end
    return result
end

function _besselix_asymptotic(v, a)
    T = Reactant.unwrapped_eltype(a)
    h = hypot(v, a)
    r = inv(h)
    p = v / h
    t = p * p
    # h - a = v^2/(h+a); this form avoids subtraction and v^2 overflow.
    exponent = v * p / (1 + a / h) - v * asinh(v / a)
    correction = one(a)
    rk = one(a)
    for coefficients in _BESSELIX_DEBYE
        rk = rk * r
        correction = correction + rk * evalpoly(t, T.(coefficients))
    end
    return exp(exponent) * correction / sqrt(h) / T(sqrt(2pi))
end

function _besselix_integer(n, x)
    T = Reactant.unwrapped_eltype(x)
    v = abs(oftype(x, n))
    a = abs(x)
    # Switching earlier in Float32 avoids cancellation in the series' AD
    # near the maximum of the scaled function (e.g. n=8, x=64).
    series_limit = T === Float32 ? 32 : 64
    Reactant.@trace if a == 0
        if v == 0
            result = one(x)
        elseif v == 1
            result = x / 2
        else
            result = zero(x)
        end
    elseif isinf(v)
        result = zero(x)
    elseif a <= series_limit
        result = _besselix_series(v, a)
    elseif isinf(a)
        result = zero(x)
    else
        result = _besselix_asymptotic(v, a)
    end
    Reactant.@trace if (x < 0) & isodd(n)
        result = -result
    end
    return result
end

function SpecialFunctions.besselix(
    n::Integer, x::TracedRNumber{T}
) where {T<:Union{Float32,Float64}}
    return _besselix_integer(n, x)
end

function SpecialFunctions.besselix(
    n::TracedRNumber{<:ReactantInt}, x::TracedRNumber{T}
) where {T<:Union{Float32,Float64}}
    return _besselix_integer(n, x)
end
