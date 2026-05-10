# StdTDist — Student's t with degrees of freedom `ν`, mean 0, scale 1.
# pdf(z; ν) = Γ((ν+1)/2) / (sqrt(ν π) Γ(ν/2)) · (1 + z²/ν)^(-(ν+1)/2)
#
# `ν` may be a scalar or a per-element array. `_loggamma` (Lanczos
# approximation, defined in `distributions.jl`) is used inside `lognorm`
# so this works without `SpecialFunctions`.

struct StdTDist{T,Tν,N} <: AbstractDistribution{N}
    ν::Tν
    dims::Dims{N}
end
StdTDist(ν::Number) = StdTDist{typeof(ν),typeof(ν),0}(ν, ())
StdTDist(ν::Number, dims::Dims{N}) where {N} = StdTDist{typeof(ν),typeof(ν),N}(ν, dims)
StdTDist(ν::Number, dims::Int...) = StdTDist(ν, dims)
function StdTDist(ν::AbstractArray{<:Number,N}) where {N}
    return StdTDist{eltype(ν),typeof(ν),N}(ν, size(ν))
end

Base.size(d::StdTDist) = d.dims
Base.length(d::StdTDist) = prod(d.dims)
Base.eltype(::StdTDist{T}) where {T} = T

# ----- unnormalised log-pdf ----------------------------------------------

@inline function _unnormed_kernel(d::StdTDist, z)
    ν = d.ν
    return -((ν + one(ν)) / 2) * log1p(z * z / ν)
end
@inline function _unnormed_kernel_sum(d::StdTDist, z)
    ν = d.ν
    sq = abs2.(z)
    log_terms = log1p.(sq ./ ν)
    return -sum(((ν .+ 1) ./ 2) .* log_terms)
end

unnormed_logpdf(d::StdTDist{T,<:Number,0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(d::StdTDist{T,Tν,N}, x::AbstractArray{<:Number,N}) where {T,Tν,N}
    return _unnormed_kernel_sum(d, x)
end

# ----- lognorm ------------------------------------------------------------
# log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(π) − ½ log(ν) per element.

@inline function _t_lognorm_per_elem(ν)
    return _loggamma((ν + 1) / 2) - _loggamma(ν / 2) - oftype(ν, log(π)) / 2 - log(ν) / 2
end
lognorm(d::StdTDist{T,<:Number}) where {T} = length(d) * _t_lognorm_per_elem(d.ν)
lognorm(d::StdTDist{T,<:AbstractArray}) where {T} = sum(_t_lognorm_per_elem, d.ν)

# ----- logpdf -------------------------------------------------------------

function logpdf(d::StdTDist{T,<:Number,0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end
function logpdf(d::StdTDist{T,Tν,N}, x::AbstractArray{<:Number,N}) where {T,Tν,N}
    size(x) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, x) + lognorm(d)
end

# ----- sampling -----------------------------------------------------------
# `Z / sqrt(Gamma(ν/2, 1) * 2 / ν)` where `Z ~ N(0, 1)`.

@inline function _sample_t(rng::AbstractRNG, ν::Real)
    z = randn(rng)
    g = _rand_gamma(rng, ν / 2)
    return z / sqrt(g * 2 / ν)
end

function Base.rand(rng::AbstractRNG, d::StdTDist{T,<:Number,0}) where {T}
    return T(_sample_t(rng, Float64(d.ν)))
end
function Random.rand!(
    rng::AbstractRNG, d::StdTDist{T,<:Number,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    νf = Float64(d.ν)
    @inbounds for i in eachindex(x)
        x[i] = _sample_t(rng, νf)
    end
    return x
end
function Random.rand!(
    rng::AbstractRNG, d::StdTDist{T,<:AbstractArray,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    @inbounds for i in eachindex(x)
        x[i] = _sample_t(rng, Float64(d.ν[i]))
    end
    return x
end

# ----- support / moments --------------------------------------------------

insupport(::StdTDist, ::Number) = true
insupport(d::StdTDist, x::AbstractArray) = size(d) == size(x)

function mean(d::StdTDist{T,<:Real,0}) where {T}
    return d.ν > 1 ? zero(T) : T(NaN)
end
function var(d::StdTDist{T,<:Real,0}) where {T}
    return d.ν > 2 ? T(d.ν / (d.ν - 2)) : T(Inf)
end
@inline _t_elemmean(ν::Real, T) = ν > 1 ? zero(T) : T(NaN)
@inline _t_elemvar(ν::Real, T) = ν > 2 ? T(ν / (ν - 2)) : T(Inf)
function mean(d::StdTDist{T,<:Real,N}) where {T,N}
    return fill(_t_elemmean(d.ν, T), size(d))
end
function var(d::StdTDist{T,<:Real,N}) where {T,N}
    return fill(_t_elemvar(d.ν, T), size(d))
end
function mean(d::StdTDist{T,<:AbstractArray,N}) where {T,N}
    return _t_elemmean.(d.ν, T)
end
function var(d::StdTDist{T,<:AbstractArray,N}) where {T,N}
    return _t_elemvar.(d.ν, T)
end

params(d::StdTDist) = (d.ν,)

# ----- cdf / quantile -----------------------------------------------------
# Like `StdInverseGamma`, the closed-form needs the regularised incomplete
# beta function which we don't ship in-tree. Calls to `cdf(::StdTDist, _)`
# / `quantile(::StdTDist, _)` raise `MethodError` until we implement
# `beta_inc` / `beta_inc_inv` in Julia.


# ----- ProbProg trait API -------------------------------------------------
# `params(d) = (ν, μ, σ)`. Sample = `μ + σ · Z / sqrt(Gamma(ν/2, 1) · 2/ν)`.

function _tdist_sampler(rng, ν, μ, σ, shape)
    νf = Float64(ν)
    sample_one = function ()
        z = randn(rng)
        g = _rand_gamma(rng, νf / 2)
        return μ + σ * z / sqrt(g * 2 / νf)
    end
    isempty(shape) && return sample_one()
    out = Array{Float64}(undef, shape...)
    @inbounds for i in eachindex(out)
        out[i] = sample_one()
    end
    return out
end
function _tdist_logpdf(x, ν, μ, σ, _shape)
    z = (x .- μ) ./ σ
    lognorm_per =
        _loggamma.((ν .+ 1) ./ 2) .- _loggamma.(ν ./ 2) .-
        log(π) ./ 2 .- log.(ν) ./ 2
    return sum(lognorm_per .- ((ν .+ 1) ./ 2) .* log1p.(z .* z ./ ν) .- log.(σ))
end


# ----- user-facing TDist constructor -------------------------------------
# `TDist(ν[, μ, σ[, dims]])` returns an `AffineDistribution` over
# `StdTDist`. `ν` is the degrees of freedom; `μ`/`σ` shift and scale the
# standard t. The single-argument form `TDist(ν)` is equivalent to
# `TDist(ν, 0.0, 1.0)`.

TDist(ν::Number) = TDist(ν, 0.0, 1.0)
function TDist(ν::Number, μ::Number, σ::Number)
    T = float(promote_type(unwrapped_eltype(ν), unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdTDist{T,typeof(ν),0}(ν, ()), μ, σ)
end
function TDist(ν::Number, μ::Number, σ::Number, dims::Dims{N}) where {N}
    T = float(promote_type(unwrapped_eltype(ν), unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdTDist{T,typeof(ν),N}(ν, dims), μ, σ)
end
function TDist(ν::Number, μ::Number, σ::Number, dims::Int...)
    return TDist(ν, μ, σ, dims)
end
function TDist(
    ν::AbstractArray{<:Number,N}, μ::AbstractArray{<:Number,N}, σ::AbstractArray{<:Number,N}
) where {N}
    size(ν) == size(μ) == size(σ) ||
        throw(DimensionMismatch("TDist: ν, μ, σ must have the same shape"))
    T = float(promote_type(unwrapped_eltype(ν), unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdTDist{T,typeof(ν),N}(ν, size(ν)), μ, σ)
end
function TDist(ν::AbstractArray{<:Number,N}, μ::Number, σ::Number) where {N}
    T = float(promote_type(unwrapped_eltype(ν), unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdTDist{T,typeof(ν),N}(ν, size(ν)), μ, σ)
end
function TDist(
    ν::Number, μ::AbstractArray{<:Number,N}, σ::AbstractArray{<:Number,N}
) where {N}
    size(μ) == size(σ) ||
        throw(DimensionMismatch("TDist: μ and σ must have the same shape"))
    T = float(promote_type(unwrapped_eltype(ν), unwrapped_eltype(μ), unwrapped_eltype(σ)))
    return AffineDistribution(StdTDist{T,typeof(ν),N}(ν, size(μ)), μ, σ)
end


# ----- per-base specialisations on `AffineDistribution{<:StdTDist}` ------

params(d::AffineDistribution{<:StdTDist}) =
    (d.base.ν, d.transform.loc, d.transform.scale)

sampler(::Type{<:AffineDistribution{<:StdTDist}}) = _tdist_sampler
logpdf_fn(::Type{<:AffineDistribution{<:StdTDist}}) = _tdist_logpdf
support(::Type{<:AffineDistribution{<:StdTDist}}) = RealSupport()
