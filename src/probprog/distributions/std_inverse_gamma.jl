# StdInverseGamma — inverse-gamma with shape `α` and scale 1.
# pdf(z; α) = z^(-α-1) exp(-1/z) / Γ(α) for z > 0.
#
# `α` may be a scalar (broadcast) or an array of the same shape as the
# distribution. `_loggamma` is the in-tree Lanczos approximation defined in
# `distributions.jl` — no `SpecialFunctions` dependency required.

struct StdInverseGamma{T,Tα,N} <: AbstractDistribution{N}
    α::Tα
    dims::Dims{N}
end
StdInverseGamma(α::Number) = StdInverseGamma{typeof(α),typeof(α),0}(α, ())
function StdInverseGamma(α::Number, dims::Dims{N}) where {N}
    return StdInverseGamma{typeof(α),typeof(α),N}(α, dims)
end
StdInverseGamma(α::Number, dims::Int...) = StdInverseGamma(α, dims)
function StdInverseGamma(α::AbstractArray{<:Number,N}) where {N}
    return StdInverseGamma{eltype(α),typeof(α),N}(α, size(α))
end

Base.size(d::StdInverseGamma) = d.dims
Base.length(d::StdInverseGamma) = prod(d.dims)
Base.eltype(::StdInverseGamma{T}) where {T} = T

# ----- log-pdf split ------------------------------------------------------
# `loggamma(α)` (Lanczos approximation, in `distributions.jl`) is the
# expensive piece that lives in `lognorm`. Caching `lognorm(d)` lets a
# caller reuse it across many `logpdf` evaluations with the same `α`.

lognorm(d::StdInverseGamma{T,<:Number}) where {T} = -length(d) * _loggamma(d.α)
lognorm(d::StdInverseGamma{T,<:AbstractArray}) where {T} = -sum(_loggamma, d.α)

@inline function _unnormed_kernel(d::StdInverseGamma, z)
    α = d.α
    zsafe = ifelse(z > zero(z), z, oftype(z, 1))
    val = -(α + one(α)) * log(zsafe) - inv(zsafe)
    return ifelse(z > zero(z), val, oftype(z, -Inf))
end
@inline function _unnormed_kernel_sum(d::StdInverseGamma, z)
    α = d.α
    log_z = log.(z)
    inv_z = inv.(z)
    return -sum((α .+ 1) .* log_z) - sum(inv_z)
end

function unnormed_logpdf(d::StdInverseGamma{T,<:Number,0}, x::Number) where {T}
    return _unnormed_kernel(d, x)
end
function unnormed_logpdf(
    d::StdInverseGamma{T,Tα,N}, x::AbstractArray{<:Number,N}
) where {T,Tα,N}
    return _unnormed_kernel_sum(d, x)
end

# ----- logpdf -------------------------------------------------------------

function logpdf(d::StdInverseGamma{T,<:Number,0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function logpdf(d::StdInverseGamma{T,Tα,N}, x::AbstractArray{<:Number,N}) where {T,Tα,N}
    size(x) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, x) + lognorm(d)
end

# ----- sampling -----------------------------------------------------------
# `1 / Gamma(α, 1)` via the Marsaglia–Tsang sampler in `distributions.jl`.

function Base.rand(rng::AbstractRNG, d::StdInverseGamma{T,<:Number,0}) where {T}
    return T(inv(_rand_gamma(rng, Float64(d.α))))
end
function Random.rand!(
    rng::AbstractRNG, d::StdInverseGamma{T,<:Number,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    αf = Float64(d.α)
    @inbounds for i in eachindex(x)
        x[i] = inv(_rand_gamma(rng, αf))
    end
    return x
end
function Random.rand!(
    rng::AbstractRNG, d::StdInverseGamma{T,<:AbstractArray,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    @inbounds for i in eachindex(x)
        x[i] = inv(_rand_gamma(rng, Float64(d.α[i])))
    end
    return x
end

# ----- support / moments --------------------------------------------------

insupport(::StdInverseGamma, x::Number) = x > 0
function insupport(d::StdInverseGamma, x::AbstractArray)
    return size(d) == size(x) && all(>(0), x)
end

function mean(d::StdInverseGamma{T,<:Real,0}) where {T}
    return d.α > 1 ? T(1 / (d.α - 1)) : T(Inf)
end
function var(d::StdInverseGamma{T,<:Real,0}) where {T}
    return d.α > 2 ? T(1 / ((d.α - 1)^2 * (d.α - 2))) : T(Inf)
end
@inline _ig_elemmean(α::Real, T) = α > 1 ? T(1 / (α - 1)) : T(Inf)
@inline _ig_elemvar(α::Real, T) = α > 2 ? T(1 / ((α - 1)^2 * (α - 2))) : T(Inf)
function mean(d::StdInverseGamma{T,<:Real,N}) where {T,N}
    return fill(_ig_elemmean(d.α, T), size(d))
end
function var(d::StdInverseGamma{T,<:Real,N}) where {T,N}
    return fill(_ig_elemvar(d.α, T), size(d))
end
function mean(d::StdInverseGamma{T,<:AbstractArray,N}) where {T,N}
    return _ig_elemmean.(d.α, T)
end
function var(d::StdInverseGamma{T,<:AbstractArray,N}) where {T,N}
    return _ig_elemvar.(d.α, T)
end

params(d::StdInverseGamma) = (d.α,)

# ----- cdf / quantile -----------------------------------------------------
# Closed-form `cdf` / `quantile` for `StdInverseGamma` need the regularised
# incomplete gamma functions, which we don't currently have an in-tree
# implementation for. They're intentionally absent — calling
# `cdf(::StdInverseGamma, _)` raises `MethodError`. Add later if needed via
# a continued-fraction implementation (Numerical Recipes §6.2 or similar).


# ----- ProbProg trait API -------------------------------------------------
# `params(d) = (α, θ)`. Sample = `θ / Gamma(α, 1)` (Marsaglia–Tsang).
# Logpdf uses the in-tree `_loggamma` (Lanczos).

# Thin top-level wrappers for `Modeling.jl` — see `std_normal.jl` for the
# rationale.
_invgamma_sampler(rng, α, θ, shape::Tuple{}) = rand(rng, InverseGamma(α, θ))
_invgamma_sampler(rng, α, θ, shape::Dims) = rand(rng, InverseGamma(α, θ, shape))

_invgamma_logpdf(x::Number, α, θ, _shape) = logpdf(InverseGamma(α, θ), x)
_invgamma_logpdf(x::AbstractArray, α, θ, _shape) =
    logpdf(InverseGamma(α, θ, size(x)), x)


# ----- user-facing InverseGamma constructor ------------------------------
# `InverseGamma(α, θ[, dims])` returns an `AffineDistribution` over
# `StdInverseGamma` with `loc = 0` and `scale = θ`.

function InverseGamma(α::Number, θ::Number)
    T = float(promote_type(unwrapped_eltype(α), unwrapped_eltype(θ)))
    return AffineDistribution(StdInverseGamma{T,typeof(α),0}(α, ()), zero(θ), θ)
end
function InverseGamma(α::Number, θ::Number, dims::Dims{N}) where {N}
    T = float(promote_type(unwrapped_eltype(α), unwrapped_eltype(θ)))
    return AffineDistribution(StdInverseGamma{T,typeof(α),N}(α, dims), zero(θ), θ)
end
InverseGamma(α::Number, θ::Number, dims::Int...) = InverseGamma(α, θ, dims)
function InverseGamma(α::AbstractArray{<:Number,N}, θ::AbstractArray{<:Number,N}) where {N}
    size(α) == size(θ) ||
        throw(DimensionMismatch("InverseGamma: α and θ must have the same shape"))
    T = float(promote_type(unwrapped_eltype(α), unwrapped_eltype(θ)))
    return AffineDistribution(
        StdInverseGamma{T,typeof(α),N}(α, size(α)), zero(eltype(θ)), θ
    )
end
function InverseGamma(α::Number, θ::AbstractArray{<:Number,N}) where {N}
    T = float(promote_type(unwrapped_eltype(α), unwrapped_eltype(θ)))
    return AffineDistribution(
        StdInverseGamma{T,typeof(α),N}(α, size(θ)), zero(eltype(θ)), θ
    )
end
function InverseGamma(α::AbstractArray{<:Number,N}, θ::Number) where {N}
    T = float(promote_type(unwrapped_eltype(α), unwrapped_eltype(θ)))
    return AffineDistribution(
        StdInverseGamma{T,typeof(α),N}(α, size(α)), zero(θ), θ
    )
end


# ----- per-base specialisations on `AffineDistribution{<:StdInverseGamma}` -----

params(d::AffineDistribution{<:StdInverseGamma}) = (d.base.α, d.transform.scale)

sampler(::Type{<:AffineDistribution{<:StdInverseGamma}}) = _invgamma_sampler
logpdf_fn(::Type{<:AffineDistribution{<:StdInverseGamma}}) = _invgamma_logpdf
support(::Type{<:AffineDistribution{<:StdInverseGamma}}) = PositiveSupport()
