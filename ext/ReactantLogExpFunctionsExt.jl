module ReactantLogExpFunctionsExt

using LogExpFunctions: LogExpFunctions
using Reactant: Reactant, AnyTracedRArray, TracedRNumber, ReactantFloat
using Reactant.Ops: @opcall
using IrrationalConstants: IrrationalConstants

# xlogx: x * log(x) with handling for x = 0
function LogExpFunctions.xlogx(x::TracedRNumber)
    result = x * log(x)
    return ifelse(iszero(x), zero(result), result)
end

# xlogy: x * log(y) with handling for x = 0
function LogExpFunctions.xlogy(x::TracedRNumber, y::TracedRNumber)
    result = x * log(y)
    return ifelse(iszero(x) & !isnan(y), zero(result), result)
end

# xlog1py: x * log1p(y) with handling for x = 0
function LogExpFunctions.xlog1py(x::TracedRNumber, y::TracedRNumber)
    result = x * log1p(y)
    return ifelse(iszero(x) & !isnan(y), zero(result), result)
end

# xexpx: x * exp(x) with handling for -Inf
function LogExpFunctions.xexpx(x::TracedRNumber{<:Real})
    expx = exp(x)
    return ifelse(iszero(expx), expx, x * expx)
end

# xexpy: x * exp(y) with special handling
function LogExpFunctions.xexpy(x::TracedRNumber{<:Real}, y::TracedRNumber{<:Real})
    expy = exp(y)
    result = x * expy
    return ifelse(
        (iszero(x) & isfinite(y)) | (iszero(expy) & !isnan(x)), zero(result), result
    )
end

# logistic: sigmoid function with bounds handling
LogExpFunctions.logistic(x::TracedRNumber) = @opcall logistic(x)

# log1psq: log(1+x^2) with handling for large x
LogExpFunctions.log1psq(x::TracedRNumber{<:Real}) = log1p(abs2(x))

# log1pexp: log(1+exp(x)) with careful evaluation
@inline function _log1pexp_thresholds(x::TracedRNumber{T}) where {T<:Real}
    prec = precision(x)
    logtwo = oftype(x, IrrationalConstants.logtwo)
    x1 = (exponent(nextfloat(zero(T))) - 1) * logtwo
    x2 = -prec * logtwo
    x3 = (prec - 1) * logtwo / 2
    x4 = -x2 - log(-x2) * (1 + 1 / x2) # approximate root of e^-x == x * ϵ/2 via asymptotics of Lambert's W function
    return (x1, x2, x3, x4)
end

function LogExpFunctions.log1pexp(x::TracedRNumber{<:Real})
    x1, x2, x3, x4 = _log1pexp_thresholds(x)
    return ifelse(
        x < x1,
        zero(x),
        ifelse(
            x < x2, exp(x), ifelse(x < x3, log1p(exp(x)), ifelse(x < x4, x + exp(-x), x))
        ),
    )
end

# log1mexp: log(1 - exp(x))
function LogExpFunctions.log1mexp(x::TracedRNumber{<:Real})
    loghalf = oftype(float(x), IrrationalConstants.loghalf)
    return ifelse(x < loghalf, log1p(-exp(x)), log(-expm1(x)))
end

# logexpm1: log(exp(x) - 1)
function LogExpFunctions.logexpm1(x::TracedRNumber{<:Real})
    return ifelse(
        x <= 18.0, log(expm1(x)), ifelse(x <= 33.3, x - exp(-x), oftype(exp(-x), x))
    )
end

function LogExpFunctions.logexpm1(x::TracedRNumber{Float32})
    return ifelse(
        x <= 9.0f0, log(expm1(x)), ifelse(x <= 16.0f0, x - exp(-x), oftype(exp(-x), x))
    )
end

# log1pmx: log(1+x) - x (naive fallback for TracedRNumber)
LogExpFunctions.log1pmx(x::TracedRNumber{<:Real}) = log1p(x) - x

# logmxp1: log(x) - x + 1
function LogExpFunctions.logmxp1(x::TracedRNumber{<:Real})
    one_x = one(x)
    # For small values of x, use (log(x) + 1) - x
    # For larger values, use log1pmx(x - 1)
    return ifelse(2 * x < one_x, (log(x) + one_x) - x, LogExpFunctions.log1pmx(x - one_x))
end

# logaddexp: log(exp(x) + exp(y))
function LogExpFunctions.logaddexp(x::TracedRNumber{<:Real}, y::TracedRNumber{<:Real})
    # Compute max and diff in a branchless way
    a_lt_b = x < y
    diff = ifelse(a_lt_b, x - y, ifelse(x == y, zero(x - y), y - x))
    max_val = ifelse(a_lt_b, y, ifelse(!isnan(y), x, y))
    return max_val + LogExpFunctions.log1pexp(diff)
end

# logsubexp: log(abs(exp(x) - exp(y)))
function LogExpFunctions.logsubexp(x::TracedRNumber{<:Real}, y::TracedRNumber{<:Real})
    # Handle x == y case specially
    Δ = ifelse((x == y) & (isfinite(x) | (x < zero(x))), zero(x - y), abs(x - y))
    return max(x, y) + LogExpFunctions.log1mexp(-Δ)
end

function LogExpFunctions.softmax!(
    r::AnyTracedRArray{<:Real}, x::AnyTracedRArray{<:Real}=r; dims=:
)
    return LogExpFunctions._softmax!(r, x, dims)
end

function LogExpFunctions.softmax(x::AnyTracedRArray{<:Real}; dims=:)
    return LogExpFunctions._softmax!(similar(x, float(eltype(x))), x, dims)
end

for (T1, T2) in [(TracedRNumber, Number), (Number, TracedRNumber)]
    for binop in [:xlogy, :xlog1py, :xexpy, :logaddexp, :logsubexp]
        @eval function LogExpFunctions.$(binop)(x::$T1, y::$T2)
            T = TracedRNumber{
                promote_type(Reactant.unwrapped_eltype(x), Reactant.unwrapped_eltype(y))
            }
            return LogExpFunctions.$(binop)(
                Reactant.promote_to(T, x), Reactant.promote_to(T, y)
            )
        end
    end
end

LogExpFunctions.logit(x::TracedRNumber{<:Real}) = log(x / (one(x) - x))

function LogExpFunctions.logcosh(x::TracedRNumber{<:Real})
    abs_x = abs(x)
    return abs_x + LogExpFunctions.log1pexp(-2 * abs_x) - IrrationalConstants.logtwo
end

function LogExpFunctions.logabssinh(x::TracedRNumber{<:Real})
    abs_x = abs(x)
    return abs_x + LogExpFunctions.log1mexp(-2 * abs_x) - IrrationalConstants.logtwo
end

LogExpFunctions.log2mexp(x::TracedRNumber{<:Real}) = log1p(-expm1(x))

LogExpFunctions.softplus(x::TracedRNumber{<:Real}) = LogExpFunctions.log1pexp(x)

LogExpFunctions.invsoftplus(x::TracedRNumber{<:Real}) = LogExpFunctions.logexpm1(x)

LogExpFunctions.loglogistic(x::TracedRNumber{<:Real}) = -LogExpFunctions.log1pexp(-float(x))

LogExpFunctions.logitexp(x::TracedRNumber{<:Real}) = -LogExpFunctions.logexpm1(-float(x))

function LogExpFunctions.log1mlogistic(x::TracedRNumber{<:Real})
    return -LogExpFunctions.log1pexp(float(x))
end

LogExpFunctions.logit1mexp(x::TracedRNumber{<:Real}) = LogExpFunctions.logexpm1(-float(x))

function LogExpFunctions.logsumexp(x::AnyTracedRArray; dims=:)
    max_ = maximum(x; dims)
    return max_ .+ log.(sum(exp.(x .- max_); dims))
end

function LogExpFunctions.logsumexp!(out::AnyTracedRArray, X::AnyTracedRArray)
    out .= LogExpFunctions.logsumexp(X)
    return out
end

end
