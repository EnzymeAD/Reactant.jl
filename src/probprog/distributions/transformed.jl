# Generic transformed-distribution wrapper. If `z ~ base` and `t` is a
# bijective transform (see `transforms.jl`), then
# `TransformedDistribution(base, t)` represents the law of `y = forward(t, z)`.
#
# Mathematically: log p_y(y) = log p_z(t⁻¹(y)) − log|det J_t(t⁻¹(y))|.
#
# The `unnormed_logpdf` / `lognorm` split is dispatched on whether the
# transform's Jacobian is data-independent:
#
#   • `t::ConstantJacobianTransform` (Identity, Affine, constant compose):
#     Jacobian doesn't depend on `z`, so the per-call `logabsdetjac(t)`
#     constant goes into `lognorm(d)` and gets cached across logpdf calls.
#
#   • `t::AbstractTransform` (Log, Logit, mixed compose, …): Jacobian is
#     data-dependent, so it goes into `unnormed_logpdf(d, y)`.
#
# `AffineDistribution` is a parametric alias on top of this, so dispatch
# points like `AffineDistribution{<:StdNormal}` work transparently. Per-base
# specialisations (`params`, `cdf`/`quantile`, ProbProg trait dispatch,
# user-facing constructors like `Normal`/`LogNormal`) live alongside their
# base in the corresponding `std_*.jl` file.

"""
    TransformedDistribution(base, transform)

Represents the law of `y = forward(transform, z)` where `z ~ base`. The
transform must be bijective with a known inverse and Jacobian — see
[`AbstractTransform`](@ref).
"""
struct TransformedDistribution{D<:AbstractDistribution,T<:AbstractTransform,N} <:
       AbstractDistribution{N}
    base::D
    transform::T
end
function TransformedDistribution(
    base::AbstractDistribution{N}, transform::AbstractTransform
) where {N}
    return TransformedDistribution{typeof(base),typeof(transform),N}(base, transform)
end

Base.size(d::TransformedDistribution) = size(d.base)
Base.length(d::TransformedDistribution) = length(d.base)
function Base.eltype(d::TransformedDistribution)
    return promote_type(_transform_eltype(d.transform), eltype(d.base))
end
@inline _transform_eltype(::IdentityTransform) = Union{}
@inline _transform_eltype(t::AffineTransform) = promote_type(eltype(t.loc), eltype(t.scale))
@inline _transform_eltype(::LogTransform) = Union{}
@inline _transform_eltype(::LogitTransform) = Union{}
@inline _transform_eltype(t::Union{ComposeTransform,ComposeConst}) =
    promote_type(_transform_eltype(t.inner), _transform_eltype(t.outer))

function Base.show(io::IO, d::TransformedDistribution)
    print(io, "TransformedDistribution(")
    print(io, "base=", nameof(typeof(d.base)))
    print(io, ", transform=", d.transform)
    sz = size(d)
    isempty(sz) || print(io, ", size=", sz)
    return print(io, ")")
end


# ----- AffineDistribution alias + convenience constructor ---------------

"""
    AffineDistribution{D, N, Tloc, Tscale}

Type alias for `TransformedDistribution{D, AffineTransform{Tloc, Tscale}, N}`.
Constructing one with `AffineDistribution(base, loc, scale)` is identical to
`TransformedDistribution(base, AffineTransform(loc, scale))`.
"""
const AffineDistribution{D,N,Tloc,Tscale} =
    TransformedDistribution{D,AffineTransform{Tloc,Tscale},N}

function AffineDistribution(base::AbstractDistribution, loc, scale)
    if loc isa AbstractArray
        size(loc) == size(base) ||
            throw(DimensionMismatch("AffineDistribution: size(loc) must match size(base)"))
    end
    return TransformedDistribution(base, AffineTransform(loc, scale))
end


# ----- unnormed_logpdf / lognorm split ------------------------------------

# General case (data-dependent Jacobian).
function unnormed_logpdf(
    d::TransformedDistribution{D,T,0}, y::Number
) where {D,T<:AbstractTransform}
    z = inverse(d.transform, y)
    return unnormed_logpdf(d.base, z) - logabsdetjac(d.transform, z)
end
function unnormed_logpdf(
    d::TransformedDistribution{D,T,N}, y::AbstractArray{<:Number,N}
) where {D,T<:AbstractTransform,N}
    z = inverse(d.transform, y)
    return unnormed_logpdf(d.base, z) - logabsdetjac(d.transform, z)
end
lognorm(
    d::TransformedDistribution{D,T}
) where {D<:AbstractDistribution,T<:AbstractTransform} = lognorm(d.base)

# ConstantJacobianTransform — Jacobian is constant, push it into lognorm.
function unnormed_logpdf(
    d::TransformedDistribution{D,T,0}, y::Number
) where {D,T<:ConstantJacobianTransform}
    return unnormed_logpdf(d.base, inverse(d.transform, y))
end
function unnormed_logpdf(
    d::TransformedDistribution{D,T,N}, y::AbstractArray{<:Number,N}
) where {D,T<:ConstantJacobianTransform,N}
    return unnormed_logpdf(d.base, inverse(d.transform, y))
end
function lognorm(
    d::TransformedDistribution{D,T}
) where {D<:AbstractDistribution,T<:ConstantJacobianTransform}
    return lognorm(d.base) - logabsdetjac(d.transform)
end

# AffineTransform overrides — three regimes based on the (variate-dim,
# scale-storage) pair. The transform itself can't tell element-wise array
# scale from matrix-as-linear-operator (same Julia type), but the wrapper
# can because it knows N. We override `lognorm` here rather than at the
# transform level to keep that distinction local.
@inline function lognorm(
    d::AffineDistribution{D,N,Tloc,<:Number}
) where {D<:AbstractDistribution,N,Tloc}
    return lognorm(d.base) - length(d) * log(abs(d.transform.scale))
end
@inline function lognorm(
    d::AffineDistribution{D,N,Tloc,<:AbstractArray{<:Number,N}}
) where {D<:AbstractDistribution,N,Tloc}
    return lognorm(d.base) - sum(log ∘ abs, d.transform.scale)
end
@inline function lognorm(d::AffineDistribution)
    return lognorm(d.base) - first(logabsdet(d.transform.scale))
end


# ----- logpdf -------------------------------------------------------------

function logpdf(d::TransformedDistribution{D,T,0}, y::Number) where {D,T}
    return unnormed_logpdf(d, y) + lognorm(d)
end
function logpdf(
    d::TransformedDistribution{D,T,N}, y::AbstractArray{<:Number,N}
) where {D,T,N}
    size(y) == size(d) || throw(DimensionMismatch("input/distribution size mismatch"))
    return unnormed_logpdf(d, y) + lognorm(d)
end


# ----- sampling -----------------------------------------------------------

function Base.rand(rng::AbstractRNG, d::TransformedDistribution{D,T,0}) where {D,T}
    return forward(d.transform, rand(rng, d.base))
end

@inline function _transformed_rand!(rng, d::TransformedDistribution, x)
    z = similar(x, eltype(d.base))
    Random.rand!(rng, d.base, z)
    x .= forward(d.transform, z)
    return x
end
function Base.rand(rng::AbstractRNG, d::TransformedDistribution)
    x = similar(zeros(eltype(d), size(d)...))
    return _transformed_rand!(rng, d, x)
end
function Random.rand!(
    rng::AbstractRNG, d::TransformedDistribution{D,T,N}, x::AbstractArray{<:Real,N}
) where {D,T,N}
    return _transformed_rand!(rng, d, x)
end


# ----- support ------------------------------------------------------------

function insupport(d::TransformedDistribution{D,T,0}, y::Number) where {D,T}
    return insupport(d.base, inverse(d.transform, y))
end
function insupport(d::TransformedDistribution, y::AbstractArray)
    size(y) == size(d) || return false
    z = inverse(d.transform, y)
    return all(zi -> insupport(d.base, zi), z)
end


# ----- moments ------------------------------------------------------------
# Closed-form for affine; identity-passthrough for `IdentityTransform`. We
# don't ship analytic moments for general non-linear transforms.

mean(d::TransformedDistribution{D,IdentityTransform}) where {D} = mean(d.base)
var(d::TransformedDistribution{D,IdentityTransform}) where {D} = var(d.base)
std(d::TransformedDistribution{D,IdentityTransform}) where {D} = std(d.base)

mean(d::AffineDistribution{D,0,<:Number,<:Number}) where {D} =
    d.transform.loc + d.transform.scale * mean(d.base)
mean(d::AffineDistribution) =
    d.transform.loc .+ _affine_apply(d.transform.scale, mean(d.base))

var(d::AffineDistribution{D,0,<:Number,<:Number}) where {D<:AbstractDistribution} =
    d.transform.scale^2 * var(d.base)
var(
    d::AffineDistribution{D,N,Tloc,<:Number}
) where {D<:AbstractDistribution,N,Tloc} = d.transform.scale^2 .* var(d.base)
var(
    d::AffineDistribution{D,N,Tloc,<:AbstractArray{<:Number,N}}
) where {D<:AbstractDistribution,N,Tloc} = d.transform.scale .^ 2 .* var(d.base)

std(d::AffineDistribution{D,0,<:Number,<:Number}) where {D} = sqrt(var(d))
std(d::AffineDistribution) = sqrt.(var(d))


# ----- generic affine cdf / quantile -------------------------------------
# Works for any base whose `_std_cdf` / `_std_quantile` is defined. The
# `forward`/`inverse` calls dispatch on `AffineTransform`, so both scalar
# and per-element scale work.

function cdf(d::AffineDistribution{D,0}, y::Number) where {D}
    return _std_cdf(d.base, inverse(d.transform, y))
end
function quantile(d::AffineDistribution{D,0}, p::Number) where {D}
    return forward(d.transform, _std_quantile(d.base, p))
end
