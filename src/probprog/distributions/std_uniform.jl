# StdUniform — uniform on `[0, 1]^N`. Already normalised (`lognorm = 0`).

struct StdUniform{T,N} <: AbstractDistribution{N}
    dims::Dims{N}
end
StdUniform(dims::Dims{N}) where {N} = StdUniform{Float64,N}(dims)
StdUniform(dims::Int...) = StdUniform(dims)
StdUniform() = StdUniform{Float64,0}(())

Base.size(d::StdUniform) = d.dims
Base.length(d::StdUniform) = prod(d.dims)
Base.eltype(::StdUniform{T}) where {T} = T

# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(::StdUniform, z)
    return ifelse((z >= zero(z)) & (z <= one(z)), zero(z), oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(::StdUniform, z) = zero(eltype(z))

unnormed_logpdf(d::StdUniform{T,0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(d::StdUniform{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdUniform) = zero(eltype(d))

# ----- logpdf -------------------------------------------------------------

logpdf(d::StdUniform{T,0}, x::Number) where {T} = unnormed_logpdf(d, x) + lognorm(d)
function logpdf(d::StdUniform{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
    size(x) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, x) + lognorm(d)
end

# ----- sampling -----------------------------------------------------------

Base.rand(rng::AbstractRNG, ::StdUniform{T,0}) where {T} = rand(rng, T)
function Base.rand(rng::AbstractRNG, d::StdUniform{T,N}) where {T,N}
    return rand(rng, T, size(d)...)
end
function Random.rand!(
    rng::AbstractRNG, ::StdUniform{T,N}, x::AbstractArray{<:Real,N}
) where {T,N}
    return rand!(rng, x)
end

# ----- support / moments --------------------------------------------------

insupport(::StdUniform, x::Number) = (0 <= x <= 1)
function insupport(d::StdUniform, x::AbstractArray)
    return size(d) == size(x) && all(xi -> 0 <= xi <= 1, x)
end

mean(::StdUniform{T,0}) where {T} = T(0.5)
var(::StdUniform{T,0}) where {T} = T(1) / T(12)
std(::StdUniform{T,0}) where {T} = sqrt(var(StdUniform{T,0}(())))
mean(d::StdUniform) = fill(eltype(d)(0.5), size(d))
var(d::StdUniform) = fill(eltype(d)(1) / eltype(d)(12), size(d))
std(d::StdUniform) = fill(sqrt(eltype(d)(1) / eltype(d)(12)), size(d))

params(::StdUniform) = ()

# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdUniform, x) = clamp(x, zero(x), one(x))
@inline _std_quantile(::StdUniform, p) = p

cdf(d::StdUniform{T,0}, x::Number) where {T} = _std_cdf(d, x)
quantile(d::StdUniform{T,0}, p::Number) where {T} = _std_quantile(d, p)


# ----- ProbProg trait API -------------------------------------------------

# Thin top-level wrappers for `Modeling.jl` — see `std_normal.jl` for the
# rationale. The dispatched logpdf returns -Inf outside [a, b] (the old
# hand-rolled trait helper missed that check).
_uniform_sampler(rng, a, b, shape::Tuple{}) = rand(rng, Uniform(a, b))
_uniform_sampler(rng, a, b, shape::Dims) = rand(rng, Uniform(a, b, shape))

_uniform_logpdf(x::Number, a, b, _shape) = logpdf(Uniform(a, b), x)
_uniform_logpdf(x::AbstractArray, a, b, _shape) = logpdf(Uniform(a, b, size(x)), x)


# ----- user-facing Uniform constructor -----------------------------------
# `Uniform(a, b[, dims])` returns an `AffineDistribution` over `StdUniform`
# with `loc = a` and `scale = b - a`.

Uniform() = Uniform(0.0, 1.0)
function Uniform(a::Number, b::Number)
    T = float(promote_type(unwrapped_eltype(a), unwrapped_eltype(b)))
    return AffineDistribution(StdUniform{T,0}(()), a, b - a)
end
function Uniform(a::Number, b::Number, dims::Dims{N}) where {N}
    T = float(promote_type(unwrapped_eltype(a), unwrapped_eltype(b)))
    return AffineDistribution(StdUniform{T,N}(dims), a, b - a)
end
Uniform(a::Number, b::Number, dims::Int...) = Uniform(a, b, dims)
function Uniform(a::AbstractArray{<:Number,N}, b::AbstractArray{<:Number,N}) where {N}
    size(a) == size(b) ||
        throw(DimensionMismatch("Uniform: a and b must have the same shape"))
    T = float(promote_type(unwrapped_eltype(a), unwrapped_eltype(b)))
    return AffineDistribution(StdUniform{T,N}(size(a)), a, b .- a)
end
function Uniform(a::Number, b::AbstractArray{<:Number,N}) where {N}
    T = float(promote_type(unwrapped_eltype(a), unwrapped_eltype(b)))
    return AffineDistribution(StdUniform{T,N}(size(b)), a, b .- a)
end
function Uniform(a::AbstractArray{<:Number,N}, b::Number) where {N}
    T = float(promote_type(unwrapped_eltype(a), unwrapped_eltype(b)))
    return AffineDistribution(StdUniform{T,N}(size(a)), a, b .- a)
end


# ----- per-base specialisations on `AffineDistribution{<:StdUniform}` ----

params(d::AffineDistribution{<:StdUniform}) =
    (d.transform.loc, d.transform.loc .+ d.transform.scale)

sampler(::Type{<:AffineDistribution{<:StdUniform}}) = _uniform_sampler
logpdf_fn(::Type{<:AffineDistribution{<:StdUniform}}) = _uniform_logpdf
support(::Type{<:AffineDistribution{<:StdUniform}}) = RealSupport()
