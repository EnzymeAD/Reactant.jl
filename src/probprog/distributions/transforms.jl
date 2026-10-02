# Bijective transforms used by `TransformedDistribution`.
#
# A transform `t` represents a bijection `f` with the interface
#
#   forward(t, z)      → y = f(z)
#   inverse(t, y)      → z = f⁻¹(y)
#   logabsdetjac(t, z) → log |det J_f(z)|             (data-dependent)
#
# Plus the marker abstract type `ConstantJacobianTransform` for
# transforms whose Jacobian doesn't depend on `z` (affine, identity,
# compositions thereof). For those subtypes there's also a no-`z`
# overload `logabsdetjac(t)` returning the constant value, which lets
# `TransformedDistribution.lognorm` cache it.


# TODO reuse some of the InverseFunctions and ChangeOfVariables machinery. 
# Again I don't want to make Reactant depend on those packages so when Impulse gets separated
# we should move this stuff
"""
    AbstractTransform

Abstract supertype for bijective transforms used to build
[`TransformedDistribution`](@ref). Concrete subtypes implement
`forward`, `inverse`, and `logabsdetjac`.
"""
abstract type AbstractTransform end

"""
    ConstantJacobianTransform <: AbstractTransform

Marker for transforms whose `log|det J|` is data-independent. Subtypes
must additionally implement `logabsdetjac(t)` (no `z` argument). This
lets `TransformedDistribution` route the Jacobian into the cacheable
`lognorm` instead of `unnormed_logpdf`.
"""
abstract type ConstantJacobianTransform <: AbstractTransform end

function forward end
function inverse end
function logabsdetjac end

# Default for any ConstantJacobianTransform: the no-z method must be
# defined; the data-dependent one just defers to it.
@inline logabsdetjac(t::ConstantJacobianTransform, _) = logabsdetjac(t)


# ----- IdentityTransform -------------------------------------------------

"""
    IdentityTransform()

`f(z) = z`. Useful as a no-op wrapper or as the neutral element of
`ComposeTransform`.
"""
struct IdentityTransform <: ConstantJacobianTransform end

@inline forward(::IdentityTransform, z) = z
@inline inverse(::IdentityTransform, y) = y
@inline logabsdetjac(::IdentityTransform) = false   # exact zero, additive identity


# ----- AffineTransform ---------------------------------------------------
# `f(z) = loc + A · z` where `A` is one of:
#   * `Number`                                   — scalar diagonal
#   * `AbstractArray{<:Number, N}` matching the
#     variate dimension N                        — element-wise diagonal
#   * anything else (`AbstractMatrix`, `LinearMap`,
#     FFT plan, ...)                             — generic linear operator
#     using `*`, `\`, `logabsdet`.

"""
    AffineTransform(loc, scale)

Affine bijection `f(z) = loc + scale · z`. `loc` may be a `Number` or an
`AbstractArray` matching the variate's shape. `scale` is interpreted as
element-wise when its dimensionality matches the variate, and as a
generic linear operator (`*` / `\\` / `logabsdet`) otherwise — so a
`Cholesky` factor, an FFT plan, or any custom `LinearMap` works.
"""
struct AffineTransform{Tloc,Tscale} <: ConstantJacobianTransform
    loc::Tloc
    scale::Tscale
end

# `A * z` (forward)
@inline _affine_apply(A::Number, z) = A * z
@inline _affine_apply(
    A::AbstractArray{<:Number,N}, z::AbstractArray{<:Number,N}
) where {N} = A .* z
@inline _affine_apply(A, z) = A * z   # fallback: matrix-vector / linear map

# `A \ y` (inverse)
@inline _affine_unapply(A::Number, y) = y / A
@inline _affine_unapply(
    A::AbstractArray{<:Number,N}, y::AbstractArray{<:Number,N}
) where {N} = y ./ A
@inline _affine_unapply(A, y) = A \ y

@inline forward(t::AffineTransform, z) = t.loc .+ _affine_apply(t.scale, z)
@inline inverse(t::AffineTransform, y) = _affine_unapply(t.scale, y .- t.loc)

# `log |det A|`. Without the variate length we can't handle the scalar
# case (where `det = scale^n`); `TransformedDistribution.lognorm`
# dispatches on the wrapper to pass that length through.
@inline logabsdetjac(t::AffineTransform{<:Any,<:AbstractArray{<:Number}}) =
    sum(log ∘ abs, t.scale)
@inline logabsdetjac(t::AffineTransform) = first(logabsdet(t.scale))

# Specialised: scalar scale needs the variate length to give the right
# result. We expose a 2-arg form taking the length explicitly.
@inline _affine_logabsdet(t::AffineTransform{<:Any,<:Number}, n::Integer) =
    n * log(abs(t.scale))
@inline _affine_logabsdet(t::AffineTransform, _) = logabsdetjac(t)


# ----- LogTransform ------------------------------------------------------
# `f(z) = exp(z)`. Bijection R → R₊. Element-wise on arrays.
# `J_f(z) = exp(z) = f(z)`, so `log|det J_f(z)| = sum(z)` for arrays
# (and `z` for scalars). Data-dependent.

"""
    LogTransform()

`f(z) = exp(z)`, the bijection from ℝ to ℝ₊. Wrapping a `Normal` base in
`LogTransform` gives `LogNormal`.
"""
struct LogTransform <: AbstractTransform end

@inline forward(::LogTransform, z) = exp.(z)
@inline forward(::LogTransform, z::Number) = exp(z)
@inline inverse(::LogTransform, y) = log.(y)
@inline inverse(::LogTransform, y::Number) = log(y)

@inline logabsdetjac(::LogTransform, z::Number) = z
@inline logabsdetjac(::LogTransform, z::AbstractArray) = sum(z)


# ----- LogitTransform ----------------------------------------------------
# `f(z) = σ(z) = 1 / (1 + exp(-z))`. Bijection R → (0, 1). Element-wise.
# Using the numerically stable form `log σ'(z) = -softplus(z) - softplus(-z)`
# where `softplus(x) = log1p(exp(x))`.

"""
    LogitTransform()

`f(z) = σ(z) = 1 / (1 + e⁻ᶻ)`, the bijection from ℝ to (0, 1). Wrapping
a `Normal` base in `LogitTransform` gives `LogitNormal`.
"""
struct LogitTransform <: AbstractTransform end

# TODO when we separate impulse we really should use LogExpFunctionsExt for this stuff. 
@inline _sigmoid(z) = inv(one(z) + exp(-z))
@inline _logit(y) = log(y / (one(y) - y))
@inline _softplus(x) = log1p(exp(x))

@inline forward(::LogitTransform, z::Number) = _sigmoid(z)
@inline forward(::LogitTransform, z::AbstractArray) = _sigmoid.(z)
@inline inverse(::LogitTransform, y::Number) = _logit(y)
@inline inverse(::LogitTransform, y::AbstractArray) = _logit.(y)

# log σ'(z) = -softplus(z) - softplus(-z)  (numerically stable; equals -2·softplus(|z|) - |z| up to sign)
@inline _logit_logjac(z) = -_softplus(z) - _softplus(-z)
@inline logabsdetjac(::LogitTransform, z::Number) = _logit_logjac(z)
@inline logabsdetjac(::LogitTransform, z::AbstractArray) = sum(_logit_logjac, z)


# ----- ComposeTransform --------------------------------------------------
# `f(z) = outer(inner(z))`. Constant-Jacobian when both halves are
# constant-Jacobian; otherwise data-dependent.

"""
    ComposeTransform(inner, outer)

The composition `f(z) = outer(inner(z))`. If `inner` and `outer` are both
[`ConstantJacobianTransform`](@ref)s the composition is too (so its
`logabsdetjac` is data-independent and gets cached in `lognorm`).
Otherwise the composition is data-dependent.
"""
struct ComposeTransform{Tin,Tout} <: AbstractTransform
    inner::Tin
    outer::Tout
end

# Constant-Jacobian when *both* halves are.
struct ComposeConst{Tin<:ConstantJacobianTransform,Tout<:ConstantJacobianTransform} <:
       ConstantJacobianTransform
    inner::Tin
    outer::Tout
end

# Single entry point that picks the right struct based on the transforms'
# constant-ness.
ComposeTransform(inner::ConstantJacobianTransform, outer::ConstantJacobianTransform) =
    ComposeConst(inner, outer)

# Convenience: `t1 ∘ t2 = ComposeTransform(t2, t1)` (math composition order)
import Base: ∘
∘(outer::AbstractTransform, inner::AbstractTransform) = ComposeTransform(inner, outer)

@inline forward(t::Union{ComposeTransform,ComposeConst}, z) =
    forward(t.outer, forward(t.inner, z))
@inline inverse(t::Union{ComposeTransform,ComposeConst}, y) =
    inverse(t.inner, inverse(t.outer, y))

# Data-dependent: chain rule sums the per-link Jacobians.
@inline function logabsdetjac(t::ComposeTransform, z)
    zinner = forward(t.inner, z)
    return logabsdetjac(t.inner, z) + logabsdetjac(t.outer, zinner)
end

# Constant-Jacobian compose: just sum the constants.
@inline logabsdetjac(t::ComposeConst) = logabsdetjac(t.inner) + logabsdetjac(t.outer)


# ----- support kind dispatch ---------------------------------------------
# Each transform reports the *target* support of `f(z)`. Affine and
# Identity preserve the base's support; Log forces positive; Logit forces
# unit-interval. `nothing` means "use the base distribution's support".
# `TransformedDistribution.support` calls this and falls back to the
# base when the transform doesn't override.

transform_support(::IdentityTransform) = nothing
transform_support(::AffineTransform) = nothing
transform_support(::LogTransform) = PositiveSupport()
transform_support(::LogitTransform) = UnitIntervalSupport()
function transform_support(t::Union{ComposeTransform,ComposeConst})
    out = transform_support(t.outer)
    return isnothing(out) ? transform_support(t.inner) : out
end
