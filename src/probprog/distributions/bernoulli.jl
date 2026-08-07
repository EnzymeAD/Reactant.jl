# Bernoulli — discrete distribution on {0, 1}, logits-parameterised:
#   p(y; ℓ) = σ(ℓ)^y · (1 − σ(ℓ))^(1−y)  for y ∈ {0, 1}
# where σ(ℓ) = 1 / (1 + e^−ℓ). The logits parameterisation matches the old
# (pre-rewrite) API and avoids the σ(ℓ) numerical-precision trap near 0/1.
#
# Bernoulli is discrete so the `AffineDistribution`/`TransformedDistribution`
# machinery doesn't apply — there's no real-valued bijection from a
# unit-rate base. It lives standalone, like the std bases.

struct Bernoulli{T,Tlogits,N} <: AbstractDistribution{N}
    logits::Tlogits
    dims::Dims{N}
end
function Bernoulli(logits::Number)
    T = float(unwrapped_eltype(logits))
    return Bernoulli{T,typeof(logits),0}(logits, ())
end
function Bernoulli(logits::Number, dims::Dims{N}) where {N}
    T = float(unwrapped_eltype(logits))
    return Bernoulli{T,typeof(logits),N}(logits, dims)
end
Bernoulli(logits::Number, dims::Int...) = Bernoulli(logits, dims)
function Bernoulli(logits::AbstractArray{<:Number,N}) where {N}
    T = float(unwrapped_eltype(logits))
    return Bernoulli{T,typeof(logits),N}(logits, size(logits))
end

Base.size(d::Bernoulli) = d.dims
Base.length(d::Bernoulli) = prod(d.dims)
Base.eltype(::Bernoulli{T}) where {T} = T


# ----- log-pdf ------------------------------------------------------------
# logp(y; ℓ) = y·ℓ − log(1 + e^ℓ) = y·ℓ − softplus(ℓ).
# Use the numerically stable form `softplus(ℓ) = max(ℓ, 0) + log1p(e^−|ℓ|)`
# so we don't blow up when |ℓ| is large.

@inline _softplus_stable(ℓ) = max(ℓ, zero(ℓ)) + log1p(exp(-abs(ℓ)))
@inline _bernoulli_kernel(y, ℓ) = y * ℓ - _softplus_stable(ℓ)

# Bernoulli has no separable lognorm (the normaliser depends on logits per
# element), so we just put the full log-density in `unnormed_logpdf` and
# leave `lognorm` as zero. Callers that compose `unnormed_logpdf + lognorm`
# still get the right answer.

function unnormed_logpdf(d::Bernoulli{T,<:Number,0}, y::Number) where {T}
    return _bernoulli_kernel(y, d.logits)
end
function unnormed_logpdf(
    d::Bernoulli{T,<:Number,N}, y::AbstractArray{<:Number,N}
) where {T,N}
    return sum(_bernoulli_kernel.(y, d.logits))
end
function unnormed_logpdf(
    d::Bernoulli{T,<:AbstractArray,N}, y::AbstractArray{<:Number,N}
) where {T,N}
    return sum(_bernoulli_kernel.(y, d.logits))
end

@inline lognorm(::Bernoulli) = false   # 0 in the additive-identity sense

function logpdf(d::Bernoulli{T,<:Number,0}, y::Number) where {T}
    return unnormed_logpdf(d, y)
end
function logpdf(
    d::Bernoulli{T,Tlogits,N}, y::AbstractArray{<:Number,N}
) where {T,Tlogits,N}
    size(y) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, y)
end


# ----- sampling -----------------------------------------------------------
# `y = (U < σ(ℓ))` with `U ~ Uniform(0, 1)`. We sample `U` and convert to
# eltype(d) so the result is `Float64`-compatible (matches the old API).

@inline _bernoulli_prob(ℓ) = inv(one(ℓ) + exp(-ℓ))

function Base.rand(rng::AbstractRNG, d::Bernoulli{T,<:Number,0}) where {T}
    return T(rand(rng) < _bernoulli_prob(d.logits))
end
function Base.rand(rng::AbstractRNG, d::Bernoulli{T,<:Number,N}) where {T,N}
    p = _bernoulli_prob(d.logits)
    u = rand(rng, T, size(d)...)
    return T.(u .< p)
end
function Base.rand(rng::AbstractRNG, d::Bernoulli{T,<:AbstractArray,N}) where {T,N}
    p = _bernoulli_prob.(d.logits)
    u = rand(rng, T, size(d)...)
    return T.(u .< p)
end

function Random.rand!(
    rng::AbstractRNG, d::Bernoulli{T,<:Number,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    p = _bernoulli_prob(d.logits)
    @inbounds for i in eachindex(x)
        x[i] = rand(rng) < p
    end
    return x
end
function Random.rand!(
    rng::AbstractRNG, d::Bernoulli{T,<:AbstractArray,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    @inbounds for i in eachindex(x)
        x[i] = rand(rng) < _bernoulli_prob(d.logits[i])
    end
    return x
end


# ----- support / moments --------------------------------------------------

insupport(::Bernoulli, y::Number) = (y == 0) | (y == 1)
function insupport(d::Bernoulli, y::AbstractArray)
    return size(d) == size(y) && all(yi -> (yi == 0) | (yi == 1), y)
end

mean(d::Bernoulli{T,<:Number,0}) where {T} = T(_bernoulli_prob(d.logits))
function mean(d::Bernoulli{T,<:Number,N}) where {T,N}
    return fill(T(_bernoulli_prob(d.logits)), size(d))
end
mean(d::Bernoulli{T,<:AbstractArray}) where {T} = T.(_bernoulli_prob.(d.logits))

function var(d::Bernoulli{T,<:Number,0}) where {T}
    p = _bernoulli_prob(d.logits)
    return T(p * (one(p) - p))
end
function var(d::Bernoulli{T,<:Number,N}) where {T,N}
    p = _bernoulli_prob(d.logits)
    return fill(T(p * (one(p) - p)), size(d))
end
function var(d::Bernoulli{T,<:AbstractArray}) where {T}
    p = _bernoulli_prob.(d.logits)
    return T.(p .* (one.(p) .- p))
end

std(d::Bernoulli) = sqrt.(var(d))

params(d::Bernoulli) = (d.logits,)


# ----- ProbProg trait API -------------------------------------------------
# Thin top-level wrappers — see `std_normal.jl` for the rationale.

_bernoulli_sampler(rng, logits, shape::Tuple{}) = rand(rng, Bernoulli(logits))
_bernoulli_sampler(rng, logits, shape::Dims) = rand(rng, Bernoulli(logits, shape))

_bernoulli_logpdf(y::Number, logits, _shape) = logpdf(Bernoulli(logits), y)
_bernoulli_logpdf(y::AbstractArray, logits, _shape) =
    logpdf(Bernoulli(logits, size(y)), y)

sampler(::Type{<:Bernoulli}) = _bernoulli_sampler
logpdf_fn(::Type{<:Bernoulli}) = _bernoulli_logpdf
# Bernoulli's value space is the discrete set {0, 1}; the support attribute
# is consumed by gradient-based MCMC kernels, which don't apply to discrete
# distributions. We tag it `RealSupport()` to match the old API and let
# gradient samplers ignore the parameter.
support(::Type{<:Bernoulli}) = RealSupport()
