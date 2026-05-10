# StdNormal — standard zero-mean unit-variance normal of arbitrary shape.

struct StdNormal{T,N} <: AbstractDistribution{N}
    dims::Dims{N}
end
StdNormal(d::Dims{N}) where {N} = StdNormal{Float64,N}(d)
StdNormal(d::Int...) = StdNormal(d)

Base.size(d::StdNormal) = d.dims
Base.length(d::StdNormal) = prod(d.dims)
Base.eltype(::StdNormal{T}) where {T} = T

# ----- log-pdf split ------------------------------------------------------

@inline _unnormed_kernel(::StdNormal, z) = -z * z / 2

# `sum(abs2, z)` is non-allocating on CPU and Reactant supports it.
@inline _unnormed_kernel_sum(::StdNormal, z) = -sum(abs2, z) / 2

unnormed_logpdf(d::StdNormal{T,0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(d::StdNormal{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdNormal) = -length(d) * oftype(zero(eltype(d)), log(2π) / 2)

# ----- logpdf -------------------------------------------------------------
# No `Distributions.jl` ambiguity to break, so a single `<:Number` overload
# is enough — there's no fallback hierarchy to clash with.

logpdf(d::StdNormal{T,0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)
function logpdf(d::StdNormal{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    size(x) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, x) + lognorm(d)
end

# ----- sampling -----------------------------------------------------------

Base.rand(rng::AbstractRNG, ::StdNormal{T,0}) where {T} = T(randn(rng))
function Base.rand(rng::AbstractRNG, d::StdNormal{T,N}) where {T,N}
    return randn(rng, T, size(d)...)
end
function Random.rand!(
    rng::AbstractRNG, ::StdNormal{T,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    return randn!(rng, x)
end

# ----- support / moments --------------------------------------------------

insupport(::StdNormal, ::Number) = true
insupport(d::StdNormal, x::AbstractArray) = size(d) == size(x)

mean(::StdNormal{T,0}) where {T} = zero(T)
var(::StdNormal{T,0}) where {T} = one(T)
std(::StdNormal{T,0}) where {T} = one(T)
mean(d::StdNormal) = zeros(eltype(d), size(d))
var(d::StdNormal) = ones(eltype(d), size(d))
std(d::StdNormal) = ones(eltype(d), size(d))

params(::StdNormal) = ()

# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdNormal, x) = (one(x) + _erf(x / sqrt(oftype(x, 2)))) / 2
@inline _std_quantile(::StdNormal, p) = sqrt(oftype(p, 2)) * _erfinv(2 * p - one(p))

cdf(d::StdNormal{T,0}, x::Number) where {T} = _std_cdf(d, x)
quantile(d::StdNormal{T,0}, p::Number) where {T} = _std_quantile(d, p)


# ----- ProbProg trait API -------------------------------------------------
# Used by `Modeling.jl`'s `sample(rng, dist::D)`. The shape argument is
# appended at the call site (since our `params(d)` is shape-free).

function _normal_sampler(rng, μ, σ, shape)
    isempty(shape) && return μ + σ * randn(rng)
    return μ .+ σ .* randn(rng, shape...)
end
function _normal_logpdf(x, μ, σ, _shape)
    z = (x .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ))
end


# ----- user-facing Normal constructor ------------------------------------
# `Normal(μ, σ[, dims])` returns an `AffineDistribution` over `StdNormal`.
# `T` is the storage type for the base — we strip Reactant tracer wrappers
# via `unwrapped_eltype` and promote `Int → Float64` via `float`, so the
# base sampler `randn(rng, T)` always sees a concrete `AbstractFloat`. The
# user's `μ`/`σ` (possibly `TracedRNumber`) are kept verbatim on the affine
# transform so the trace flows through.

function Normal(μ::Number, σ::Number)
    T = float(promote_type(unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdNormal{T,0}(()), μ, σ)
end
function Normal(μ::Number, σ::Number, dims::Dims{N}) where {N}
    T = float(promote_type(unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdNormal{T,N}(dims), μ, σ)
end
Normal(μ::Number, σ::Number, dims::Int...) = Normal(μ, σ, dims)
function Normal(μ::AbstractArray{<:Number,N}, σ::AbstractArray{<:Number,N}) where {N}
    size(μ) == size(σ) ||
        throw(DimensionMismatch("Normal: μ and σ must have the same shape"))
    T = float(promote_type(unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdNormal{T,N}(size(μ)), μ, σ)
end
function Normal(μ::Number, σ::AbstractArray{<:Number,N}) where {N}
    T = float(promote_type(unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdNormal{T,N}(size(σ)), μ, σ)
end
function Normal(μ::AbstractArray{<:Number,N}, σ::Number) where {N}
    T = float(promote_type(unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdNormal{T,N}(size(μ)), μ, σ)
end


# ----- per-base specialisations on `AffineDistribution{<:StdNormal}` -----

params(d::AffineDistribution{<:StdNormal}) = (d.transform.loc, d.transform.scale)

sampler(::Type{<:AffineDistribution{<:StdNormal}}) = _normal_sampler
logpdf_fn(::Type{<:AffineDistribution{<:StdNormal}}) = _normal_logpdf
support(::Type{<:AffineDistribution{<:StdNormal}}) = RealSupport()


# ----- LogNormal ---------------------------------------------------------
# `y = exp(z)` where `z ~ Normal(μ, σ)`. Built as a `LogTransform` on top
# of an affine-Normal base so dispatch sees it as
# `TransformedDistribution{<:Normal, LogTransform}` — no new struct needed.

LogNormal(μ::Number, σ::Number) = TransformedDistribution(Normal(μ, σ), LogTransform())
function LogNormal(μ::Number, σ::Number, dims::Dims{N}) where {N}
    return TransformedDistribution(Normal(μ, σ, dims), LogTransform())
end
LogNormal(μ::Number, σ::Number, dims::Int...) = LogNormal(μ, σ, dims)

params(d::TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform}) =
    params(d.base)

# Closed-form cdf / quantile: cdf_y(y) = cdf_z(log y), quantile_y(p) = exp(quantile_z(p)).
function cdf(
    d::TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform,0}, y::Number
)
    return cdf(d.base, log(y))
end
function quantile(
    d::TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform,0}, p::Number
)
    return exp(quantile(d.base, p))
end

# Trait API for ProbProg sampling.
function _lognormal_sampler(rng, μ, σ, shape)
    isempty(shape) && return exp(μ + σ * randn(rng))
    return exp.(μ .+ σ .* randn(rng, shape...))
end
function _lognormal_logpdf(y, μ, σ, _shape)
    z = (log.(y) .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ) .- log.(y))
end
sampler(::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform}}) =
    _lognormal_sampler
logpdf_fn(::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform}}) =
    _lognormal_logpdf
support(::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogTransform}}) =
    PositiveSupport()


# ----- LogitNormal -------------------------------------------------------
# `y = σ(z)` where `z ~ Normal(μ, σ)`. Same compose-on-Normal pattern.

function LogitNormal(μ::Number, σ::Number)
    return TransformedDistribution(Normal(μ, σ), LogitTransform())
end
function LogitNormal(μ::Number, σ::Number, dims::Dims{N}) where {N}
    return TransformedDistribution(Normal(μ, σ, dims), LogitTransform())
end
LogitNormal(μ::Number, σ::Number, dims::Int...) = LogitNormal(μ, σ, dims)

params(d::TransformedDistribution{<:AffineDistribution{<:StdNormal},LogitTransform}) =
    params(d.base)

function _logitnormal_sampler(rng, μ, σ, shape)
    isempty(shape) && return _sigmoid(μ + σ * randn(rng))
    return _sigmoid.(μ .+ σ .* randn(rng, shape...))
end
function _logitnormal_logpdf(y, μ, σ, _shape)
    z = (_logit.(y) .- μ) ./ σ
    return sum(
        .-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ) .- log.(y) .- log.(one.(y) .- y)
    )
end
sampler(
    ::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogitTransform}}
) = _logitnormal_sampler
logpdf_fn(
    ::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogitTransform}}
) = _logitnormal_logpdf
support(
    ::Type{<:TransformedDistribution{<:AffineDistribution{<:StdNormal},LogitTransform}}
) = UnitIntervalSupport()
