# Reactant-friendly drop-in replacements for the scalar `Distributions.jl`
# distributions. The point of these types is to allow construction with
# `Reactant.TracedRNumber{Float64}` parameters (which are `<:Number` but not
# `<:Real`), where `Distributions.jl`'s `T<:Real` constraint blocks
# construction.
#
# Layout in this directory:
#   distributions.jl       — module, exports, shared helpers, abstract type
#   transforms.jl          — bijective transforms (Identity, Affine, Log,
#                             Logit, Compose) used by transformed.jl
#   transformed.jl         — generic `TransformedDistribution` wrapper +
#                             `AffineDistribution` alias and generic methods
#   std_normal.jl          — `StdNormal` + `Normal` / `LogNormal` /
#                             `LogitNormal` constructors and dispatches
#   std_exponential.jl     — `StdExponential` + `Exponential` constructor
#   std_uniform.jl         — `StdUniform` + `Uniform` constructor
#   std_inverse_gamma.jl   — `StdInverseGamma` + `InverseGamma` constructor
#   std_tdist.jl           — `StdTDist` + `TDist` constructor
#
# Each `std_*.jl` file owns the user-facing constructor and ProbProg
# trait dispatches for its base. `transformed.jl` is loaded first so the
# generic `AffineDistribution(...)` constructor is in scope.
#
# `loggamma`, `erf`, `erfinv` are implemented in-tree (Lanczos /
# Abramowitz / Acklam approximations) below, so the whole module is
# self-contained: no `SpecialFunctions` dependency. The implementations
# trace under Reactant naturally because they're built from `log`, `sin`,
# `exp`, `sqrt`, and polynomial arithmetic — all of which already have
# Reactant lowerings. `cdf` / `quantile` for `StdInverseGamma` and
# `StdTDist` would need `gamma_inc` / `beta_inc` and are intentionally
# omitted; calls raise `MethodError`.

module Distributions

using Random: Random
using Random: AbstractRNG, randn, randn!, randexp, randexp!

using ...Reactant: TracedRNumber, unwrapped_eltype
using ...Reactant.Ops: @opcall
using LinearAlgebra: logabsdet

export Normal, Exponential, Uniform, InverseGamma, TDist, LogNormal, LogitNormal
export StdNormal, StdExponential, StdUniform, StdInverseGamma, StdTDist
export AffineDistribution, TransformedDistribution
export AbstractTransform,
    ConstantJacobianTransform,
    IdentityTransform,
    AffineTransform,
    LogTransform,
    LogitTransform,
    ComposeTransform
export forward, inverse, logabsdetjac
export logpdf, unnormed_logpdf, lognorm
export mean, var, std, params, insupport
export cdf, quantile
export AbstractSupport,
    RealSupport,
    PositiveSupport,
    UnitIntervalSupport,
    IntervalSupport,
    GreaterThanSupport,
    LessThanSupport,
    SimplexSupport,
    LowerCholeskySupport

# ----- abstract type ------------------------------------------------------

"""
    AbstractDistribution{N}

Root of the Reactant-friendly distribution hierarchy. The type parameter `N`
is the dimensionality of a single variate (0 for scalar, 1 for vector,
2 for matrix, etc.). Mirrors `Distributions.ArrayLikeVariate{N}` semantics
without the `<:Real` parameter constraint.
"""
abstract type AbstractDistribution{N} end

# ----- public interface stubs --------------------------------------------
# Single set of canonical names that std_*.jl, transformed.jl, and external
# extensions can all `import` cleanly. Methods are added in the std files.

"""
    logpdf(d, x)

Log probability density of the distribution `d` at `x`. Equal to
`unnormed_logpdf(d, x) + lognorm(d)`.
"""
function logpdf end

"""
    unnormed_logpdf(d, x)

The `x`-dependent part of `logpdf(d, x)`. The full log-density is
`unnormed_logpdf(d, x) + lognorm(d)`.
"""
function unnormed_logpdf end

"""
    lognorm(d)

The `x`-independent log-normalisation constant of `d`.
"""
function lognorm end

function insupport end
function params end
function mean end
function var end
function std end
function cdf end
function quantile end


# ----- support kinds ------------------------------------------------------
# Type-tagged enumeration of distribution supports. Each concrete `support`
# trait returns a singleton instance — `Modeling.jl` then dispatches on the
# instance type to produce the matching MLIR `EnzymeSupportKind` attribute.
# Type-based rather than `Symbol`-based dispatch so typos are caught at
# compile time and adding a new support kind only requires defining the
# struct and its `Modeling.jl` mapping.

"""
    AbstractSupport

Base type for the singleton tags used by [`support`](@ref). Subtypes are
`RealSupport`, `PositiveSupport`, `UnitIntervalSupport`, `IntervalSupport`,
`GreaterThanSupport`, `LessThanSupport`, `SimplexSupport`,
`LowerCholeskySupport`.
"""
abstract type AbstractSupport end

struct RealSupport <: AbstractSupport end
struct PositiveSupport <: AbstractSupport end
struct UnitIntervalSupport <: AbstractSupport end
struct IntervalSupport <: AbstractSupport end
struct GreaterThanSupport <: AbstractSupport end
struct LessThanSupport <: AbstractSupport end
struct SimplexSupport <: AbstractSupport end
struct LowerCholeskySupport <: AbstractSupport end

# Trait API used by `Modeling.jl` (the probprog runtime). Each concrete
# distribution adds methods to these in its own `std_*.jl` file:
#   sampler(::Type{D})    -> (rng, params..., shape) -> samples
#   logpdf_fn(::Type{D})  -> (x, params..., shape)   -> log-density
#   support(::Type{D})    -> AbstractSupport singleton (`RealSupport()`,
#                              `PositiveSupport()`, ...)
#   bounds(::Type{D})     -> (Union{Nothing,Real}, Union{Nothing,Real})
#
# The shape argument is appended at the call site in `Modeling.jl` so our
# `params(d)` stays free of the shape tuple.

function sampler end
function logpdf_fn end
function support end
function bounds end
# ----- internal special-function shims -----------------------------------
# `loggamma`, `erf`, `erfinv` are routed straight to MLIR ops via
# `@opcall`. We assume callers are always in Reactant compile mode, so
# the eager Float64 fallbacks are gone — `_erf(0.5)` (a non-traced
# Float64) is now a `MethodError`. Tests / scripts that need an eager
# answer should `@jit` through the trace path.

_loggamma(z::TracedRNumber) = @opcall(lgamma(float(z)))
_erf(x::TracedRNumber) = @opcall(erf(float(x)))
_erfinv(p::TracedRNumber) = @opcall(erf_inv(float(p)))


# ----- Wilson–Hilferty Gamma sampler -------------------------------------
# Closed-form approximation of `Gamma(α, 1)` used by `StdInverseGamma`
# (`x = 1 / Gamma(α, 1)`) and `StdTDist`
# (`x = Z / sqrt(Gamma(ν/2, 1) · 2/ν)`, `Z ~ N(0, 1)`).
#
# `Gamma(α, 1) ≈ α · (1 − 1/(9α) + Z · √(1/(9α)))³` where `Z ~ N(0, 1)`.
# Straight-line / Reactant-traceable — no loops, no rejection. Accurate to
# a few percent for `α ≥ 3`, looser for smaller `α`. For `α < 1` we boost
# via the identity `Gamma(α) = U^(1/α) · Gamma(α + 1)`, also closed-form.
#
# Reference: Wilson & Hilferty, "The distribution of chi-square", PNAS
# 17 (1931), 684–688.

function _rand_gamma(rng::AbstractRNG, α::Number)
    # Boost α<1 to α+1, then scale by U^(1/α). `ifelse` keeps the path
    # branchless for tracing. We unconditionally draw a uniform; it's only
    # used in the boost case but the wasted op is cheap.
    u = rand(rng, typeof(α))
    α_eff = ifelse(α < one(α), α + one(α), α)
    boost = ifelse(α < one(α), u^(one(α) / α), one(α))

    z = randn(rng)
    inv_9α = one(α) / (9 * α_eff)
    base = α_eff * (one(α) - inv_9α + z * sqrt(inv_9α))^3
    return base * boost
end

include("transforms.jl")
include("transformed.jl")
include("std_normal.jl")
include("std_exponential.jl")
include("std_uniform.jl")
include("std_inverse_gamma.jl")
include("std_tdist.jl")

end # module Distributions
