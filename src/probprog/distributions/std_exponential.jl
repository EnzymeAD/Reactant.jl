# StdExponential — rate-1 exponential of arbitrary shape.
# pdf(z) = exp(-z) for z >= 0, 0 otherwise. Already normalised (`lognorm = 0`).

struct StdExponential{T,N} <: AbstractDistribution{N}
    dims::Dims{N}
end
StdExponential(dims::Dims{N}) where {N} = StdExponential{Float64,N}(dims)
StdExponential(dims::Int...) = StdExponential(dims)
StdExponential() = StdExponential{Float64,0}(())

Base.size(d::StdExponential) = d.dims
Base.length(d::StdExponential) = prod(d.dims)
Base.eltype(::StdExponential{T}) where {T} = T

# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(::StdExponential, z)
    return ifelse(z >= zero(z), -z, oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(::StdExponential, z) = -sum(z)

function unnormed_logpdf(d::StdExponential{T,0}, x::Number) where {T}
    return _unnormed_kernel(d, x)
end
function unnormed_logpdf(d::StdExponential{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdExponential) = zero(eltype(d))

# ----- logpdf -------------------------------------------------------------

function logpdf(d::StdExponential{T,0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function logpdf(d::StdExponential{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    size(x) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, x) + lognorm(d)
end

# ----- sampling -----------------------------------------------------------

Base.rand(rng::AbstractRNG, ::StdExponential{T,0}) where {T} = T(randexp(rng))
function Base.rand(rng::AbstractRNG, d::StdExponential{T,N}) where {T,N}
    return randexp(rng, T, size(d)...)
end
function Random.rand!(
    rng::AbstractRNG, ::StdExponential{T,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    return randexp!(rng, x)
end

# ----- support / moments --------------------------------------------------

insupport(::StdExponential, x::Number) = x >= 0
function insupport(d::StdExponential, x::AbstractArray)
    return size(d) == size(x) && all(>=(0), x)
end

mean(::StdExponential{T,0}) where {T} = one(T)
var(::StdExponential{T,0}) where {T} = one(T)
std(::StdExponential{T,0}) where {T} = one(T)
mean(d::StdExponential) = ones(eltype(d), size(d))
var(d::StdExponential) = ones(eltype(d), size(d))
std(d::StdExponential) = ones(eltype(d), size(d))

params(::StdExponential) = ()

# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdExponential, x) = -expm1(-x)
@inline _std_quantile(::StdExponential, p) = -log1p(-p)

cdf(d::StdExponential{T,0}, x::Number) where {T} = _std_cdf(d, x)
quantile(d::StdExponential{T,0}, p::Number) where {T} = _std_quantile(d, p)


# ----- ProbProg trait API -------------------------------------------------
# `params(d) = (θ,)` (scale). `randexp` gives unit-rate exponentials; scale
# by θ.

# Thin top-level wrappers for `Modeling.jl` — see `std_normal.jl` for the
# rationale. Math lives in the `AffineDistribution{<:StdExponential}` path.
_exponential_sampler(rng, θ, shape::Tuple{}) = rand(rng, Exponential(θ))
_exponential_sampler(rng, θ, shape::Dims) = rand(rng, Exponential(θ, shape))

_exponential_logpdf(x::Number, θ, _shape) = logpdf(Exponential(θ), x)
_exponential_logpdf(x::AbstractArray, θ, _shape) =
    logpdf(Exponential(θ, size(x)), x)


# ----- user-facing Exponential constructor -------------------------------
# `Exponential(θ[, dims])` returns an `AffineDistribution` over
# `StdExponential` with `loc = 0` and `scale = θ`.

function Exponential(θ::Number)
    T = float(unwrapped_eltype(θ))
    return AffineDistribution(StdExponential{T,0}(()), zero(θ), θ)
end
function Exponential(θ::Number, dims::Dims{N}) where {N}
    T = float(unwrapped_eltype(θ))
    return AffineDistribution(StdExponential{T,N}(dims), zero(θ), θ)
end
Exponential(θ::Number, dims::Int...) = Exponential(θ, dims)
function Exponential(θ::AbstractArray{<:Number,N}) where {N}
    T = float(unwrapped_eltype(θ))
    return AffineDistribution(StdExponential{T,N}(size(θ)), zero(eltype(θ)), θ)
end


# ----- per-base specialisations on `AffineDistribution{<:StdExponential}` -----

params(d::AffineDistribution{<:StdExponential}) = (d.transform.scale,)

sampler(::Type{<:AffineDistribution{<:StdExponential}}) = _exponential_sampler
logpdf_fn(::Type{<:AffineDistribution{<:StdExponential}}) = _exponential_logpdf
support(::Type{<:AffineDistribution{<:StdExponential}}) = PositiveSupport()
