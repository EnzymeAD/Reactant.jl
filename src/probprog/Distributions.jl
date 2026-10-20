using Random: randexp
using LinearAlgebra: cholesky, I

abstract type AbstractDistribution end
const Distribution = AbstractDistribution

abstract type AbstractTransform end

# Transform parameters are the distribution's positional operands, not attributes.
struct IdentityTransform <: AbstractTransform end
struct AffineTransform <: AbstractTransform end
struct ExpTransform <: AbstractTransform end
struct SigmoidTransform <: AbstractTransform end

xform_attr(::IdentityTransform) = "#impulse.xform<IDENTITY>"
xform_attr(::AffineTransform) = "#impulse.xform<AFFINE>"
xform_attr(::ExpTransform) = "#impulse.xform<EXP>"
xform_attr(::SigmoidTransform) = "#impulse.xform<SIGMOID>"

function _dist_spec(base::AbstractString, transforms::Vector{<:AbstractTransform})
    isempty(transforms) && return "#impulse.dist<$base>"
    inner = join(xform_attr.(transforms), ", ")
    return "#impulse.dist<$base, [$inner]>"
end
_dist_spec(base::AbstractString) = "#impulse.dist<$base>"

function dist_spec(::Type{D}) where {D<:AbstractDistribution}
    return error("dist_spec not implemented for $D")
end
dist_spec(d::AbstractDistribution) = dist_spec(typeof(d))

function support(::Type{D}) where {D<:AbstractDistribution}
    return error("support not implemented for $D")
end

function bounds(::Type{D}) where {D<:AbstractDistribution}
    return (nothing, nothing)
end

# Traced sampler: an emitted impulse.while can't reproduce its rejection-loop RNG stream.
sampler_traced(::Type{<:AbstractDistribution}) = false
sampler_traced(d::AbstractDistribution) = sampler_traced(typeof(d))

function sampler(::Type{D}) where {D<:AbstractDistribution}
    return error("sampler not implemented for $D")
end

# Returns (operands..., shape); the builder reads the operands positionally.
function params(d::AbstractDistribution)
    return error("params not implemented for $(typeof(d))")
end
dist_operands(d::AbstractDistribution) = params(d)[1:(end - 1)]
dist_shape(d::AbstractDistribution) = params(d)[end]

#==============================================================================#
# Normal Distribution
#==============================================================================#

struct Normal{Tμ,Tσ,S<:Tuple} <: AbstractDistribution
    μ::Tμ
    σ::Tσ
    shape::S

    function Normal{Tμ,Tσ,S}(μ::Tμ, σ::Tσ, shape::S) where {Tμ,Tσ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tμ,Tσ,S}(μ, σ, shape)
    end
end

Normal(μ::Tμ, σ::Tσ, shape::S) where {Tμ,Tσ,S<:Tuple} = Normal{Tμ,Tσ,S}(μ, σ, shape)
Normal() = Normal(0.0, 1.0, (1,))
Normal(μ, σ) = Normal(μ, σ, (1,))

params(d::Normal) = (d.μ, d.σ, d.shape)
support(::Type{<:Normal}) = :real
dist_spec(::Type{<:Normal}) = _dist_spec("STD_NORMAL", [AffineTransform()])

#==============================================================================#
# Uniform Distribution
#==============================================================================#

struct Uniform{Tl,Th,S<:Tuple} <: AbstractDistribution
    low::Tl
    high::Th
    shape::S

    function Uniform{Tl,Th,S}(low::Tl, high::Th, shape::S) where {Tl,Th,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tl,Th,S}(low, high, shape)
    end
end

function Uniform(low::Tl, high::Th, shape::S) where {Tl,Th,S<:Tuple}
    return Uniform{Tl,Th,S}(low, high, shape)
end
Uniform(low, high) = Uniform(low, high, (1,))

params(d::Uniform) = (d.low, d.high - d.low, d.shape)
# Bounds are operands, so HMC/NUTS/VI cannot reparameterize a Uniform site.
support(::Type{<:Uniform}) = :interval
dist_spec(::Type{<:Uniform}) = _dist_spec("STD_UNIFORM", [AffineTransform()])

#==============================================================================#
# Exponential Distribution (rate)
#==============================================================================#

struct Exponential{Tλ,S<:Tuple} <: AbstractDistribution
    λ::Tλ
    shape::S

    function Exponential{Tλ,S}(λ::Tλ, shape::S) where {Tλ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tλ,S}(λ, shape)
    end
end

Exponential(λ::Tλ, shape::S) where {Tλ,S<:Tuple} = Exponential{Tλ,S}(λ, shape)
Exponential(λ) = Exponential(λ, (1,))
Exponential() = Exponential(1.0, (1,))

params(d::Exponential) = (d.λ, d.shape)
support(::Type{<:Exponential}) = :positive
dist_spec(::Type{<:Exponential}) = _dist_spec("EXPONENTIAL")

#==============================================================================#
# HalfNormal Distribution
#==============================================================================#

struct HalfNormal{Tσ,S<:Tuple} <: AbstractDistribution
    σ::Tσ
    shape::S

    function HalfNormal{Tσ,S}(σ::Tσ, shape::S) where {Tσ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tσ,S}(σ, shape)
    end
end

HalfNormal(σ::Tσ, shape::S) where {Tσ,S<:Tuple} = HalfNormal{Tσ,S}(σ, shape)
HalfNormal(σ) = HalfNormal(σ, (1,))
HalfNormal() = HalfNormal(1.0, (1,))

params(d::HalfNormal) = (d.σ, d.shape)
support(::Type{<:HalfNormal}) = :positive
dist_spec(::Type{<:HalfNormal}) = _dist_spec("HALF_NORMAL")

#==============================================================================#
# LogNormal Distribution
#==============================================================================#

struct LogNormal{Tμ,Tσ,S<:Tuple} <: AbstractDistribution
    μ::Tμ
    σ::Tσ
    shape::S

    function LogNormal{Tμ,Tσ,S}(μ::Tμ, σ::Tσ, shape::S) where {Tμ,Tσ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tμ,Tσ,S}(μ, σ, shape)
    end
end

LogNormal(μ::Tμ, σ::Tσ, shape::S) where {Tμ,Tσ,S<:Tuple} = LogNormal{Tμ,Tσ,S}(μ, σ, shape)
LogNormal(μ, σ) = LogNormal(μ, σ, (1,))
LogNormal() = LogNormal(0.0, 1.0, (1,))

params(d::LogNormal) = (d.μ, d.σ, d.shape)
support(::Type{<:LogNormal}) = :positive
function dist_spec(::Type{<:LogNormal})
    return _dist_spec("STD_NORMAL", [AffineTransform(), ExpTransform()])
end

#==============================================================================#
# Bernoulli Distribution (logits)
#==============================================================================#

struct Bernoulli{Tp,S<:Tuple} <: AbstractDistribution
    logits::Tp
    shape::S

    function Bernoulli{Tp,S}(logits::Tp, shape::S) where {Tp,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tp,S}(logits, shape)
    end
end

Bernoulli(logits::Tp, shape::S) where {Tp,S<:Tuple} = Bernoulli{Tp,S}(logits, shape)

params(d::Bernoulli) = (d.logits, d.shape)
support(::Type{<:Bernoulli}) = :real
dist_spec(::Type{<:Bernoulli}) = _dist_spec("BERNOULLI")

#==============================================================================#
# Poisson Distribution
#==============================================================================#

struct Poisson{Tλ,S<:Tuple} <: AbstractDistribution
    rate::Tλ
    shape::S

    function Poisson{Tλ,S}(rate::Tλ, shape::S) where {Tλ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tλ,S}(rate, shape)
    end
end

Poisson(rate::Tλ, shape::S) where {Tλ,S<:Tuple} = Poisson{Tλ,S}(rate, shape)
Poisson(rate) = Poisson(rate, (1,))

params(d::Poisson) = (d.rate, d.shape)

# Knuth's algorithm (λ < 10), following jax._src.random._poisson_knuth.
function _poisson_knuth(rng, lam, shape)
    lam_shaped = Ops.fill(0.0, shape...) .+ lam
    k = Ops.fill(0.0, shape...)
    log_prod = Ops.fill(0.0, shape...)
    @trace while sum(ifelse.(log_prod .> .-lam_shaped, 1.0, 0.0)) > 0.0
        k = k .+ ifelse.(log_prod .> .-lam_shaped, 1.0, 0.0)
        u = rand(rng, size(k)...)
        log_prod = log_prod .+ log.(u)
    end
    return max.(k .- 1.0, 0.0)
end

# Hörmann's transformed rejection (λ ≥ 10), following jax._src.random._poisson_rejection.
function _poisson_rejection(rng, lam, shape)
    lam_shaped = Ops.fill(0.0, shape...) .+ lam
    log_lam = log.(lam_shaped)
    b = 0.931 .+ 2.53 .* sqrt.(lam_shaped)
    a = -0.059 .+ 0.02483 .* b
    inv_alpha = 1.1239 .+ 1.1328 ./ (b .- 3.4)
    v_r = 0.9277 .- 3.6224 ./ (b .- 2.0)

    k_out = Ops.fill(-1.0, shape...)
    accepted = Ops.fill(0.0, shape...)
    @trace while sum(1.0 .- min.(accepted, 1.0)) > 0.0
        u = rand(rng, size(k_out)...) .- 0.5
        v = rand(rng, size(k_out)...)
        u_shifted = 0.5 .- abs.(u)

        k = floor.((2.0 .* a ./ u_shifted .+ b) .* u .+ lam_shaped .+ 0.43)

        s = log.(v .* inv_alpha ./ (a ./ (u_shifted .* u_shifted) .+ b))
        t = .-lam_shaped .+ k .* log_lam .- Ops.lgamma(k .+ 1.0)

        accept1_f = ifelse.(u_shifted .>= 0.07, 1.0, 0.0) .* ifelse.(v .<= v_r, 1.0, 0.0)
        reject_f =
            min.(
                ifelse.(k .< 0.0, 1.0, 0.0) .+
                ifelse.(u_shifted .< 0.013, 1.0, 0.0) .* ifelse.(v .> u_shifted, 1.0, 0.0),
                1.0,
            )
        accept2_f = ifelse.(s .<= t, 1.0, 0.0)
        accept = min.(accept1_f .+ (1.0 .- reject_f) .* accept2_f, 1.0)

        not_yet = 1.0 .- min.(accepted, 1.0)
        new_accept = accept .* not_yet
        k_out = k_out .* (1.0 .- new_accept) .+ k .* new_accept
        accepted = accepted .+ new_accept
    end
    return max.(k_out, 0.0)
end

function _poisson_sampler(rng, rate, shape)
    rate_shaped = Ops.fill(0.0, shape...) .+ rate
    use_knuth_f = ifelse.(rate_shaped .< 10.0, 1.0, 0.0)

    lam_knuth = rate_shaped .* use_knuth_f
    lam_rejection = rate_shaped .* (1.0 .- use_knuth_f) .+ 1e5 .* use_knuth_f

    k_knuth = _poisson_knuth(rng, lam_knuth, shape)
    k_rejection = _poisson_rejection(rng, lam_rejection, shape)

    result = k_knuth .* use_knuth_f .+ k_rejection .* (1.0 .- use_knuth_f)

    nonzero_f = ifelse.(rate_shaped .> 0.0, 1.0, 0.0)
    return result .* nonzero_f
end

sampler(::Type{<:Poisson}) = _poisson_sampler
sampler_traced(::Type{<:Poisson}) = true
support(::Type{<:Poisson}) = :real
dist_spec(::Type{<:Poisson}) = _dist_spec("POISSON")

#==============================================================================#
# MultivariateNormal Distribution (mean, covariance)
#==============================================================================#

struct MultivariateNormal{Tμ,TΣ,S<:Tuple} <: AbstractDistribution
    μ::Tμ
    Σ::TΣ
    shape::S

    function MultivariateNormal{Tμ,TΣ,S}(μ::Tμ, Σ::TΣ, shape::S) where {Tμ,TΣ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tμ,TΣ,S}(μ, Σ, shape)
    end
end

function MultivariateNormal(μ::Tμ, Σ::TΣ, shape::S) where {Tμ,TΣ,S<:Tuple}
    return MultivariateNormal{Tμ,TΣ,S}(μ, Σ, shape)
end

const MultiNormal = MultivariateNormal

params(d::MultivariateNormal) = (d.μ, d.Σ, d.shape)
support(::Type{<:MultivariateNormal}) = :real
dist_spec(::Type{<:MultivariateNormal}) = _dist_spec("MVN")

#==============================================================================#
# Gamma Distribution (concentration, rate)
#==============================================================================#

struct Gamma{Tα,Tβ,S<:Tuple} <: AbstractDistribution
    concentration::Tα
    rate::Tβ
    shape::S

    function Gamma{Tα,Tβ,S}(concentration::Tα, rate::Tβ, shape::S) where {Tα,Tβ,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tα,Tβ,S}(concentration, rate, shape)
    end
end

function Gamma(concentration::Tα, rate::Tβ, shape::S) where {Tα,Tβ,S<:Tuple}
    return Gamma{Tα,Tβ,S}(concentration, rate, shape)
end
Gamma(concentration, rate) = Gamma(concentration, rate, (1,))

params(d::Gamma) = (d.concentration, d.rate, d.shape)

function _gamma_sampler(rng, concentration, rate, shape)
    # Marsaglia-Tsang, following jax.random.gamma.
    alpha = Ops.fill(0.0, shape...) .+ concentration

    alpha_orig = alpha
    boost_mask_f = ifelse.(alpha .>= 1.0, 1.0, 0.0)
    alpha = ifelse.(alpha .>= 1.0, alpha, alpha .+ 1.0)

    d = alpha .- 1.0 / 3.0
    c = (1.0 / 3.0) ./ sqrt.(d)

    V_out = Ops.fill(1.0, shape...)
    accepted = Ops.fill(0.0, shape...)

    @trace while sum(1.0 .- min.(accepted, 1.0)) > 0.0
        x = randn(rng, size(V_out)...)
        v = 1.0 .+ x .* c

        V = ifelse.(v .> 0.0, v .* v .* v, 1.0)
        v_pos_f = ifelse.(v .> 0.0, 1.0, 0.0)

        X = x .* x
        u = rand(rng, size(V_out)...)

        squeeze_f = ifelse.(u .< 1.0 .- 0.0331 .* X .* X, 1.0, 0.0)
        log_f = ifelse.(log.(u) .< X ./ 2.0 .+ d .* (1.0 .- V .+ log.(V)), 1.0, 0.0)

        accept = v_pos_f .* min.(squeeze_f .+ log_f, 1.0)

        not_yet = 1.0 .- min.(accepted, 1.0)
        new_accept = accept .* not_yet
        V_out = V_out .* (1.0 .- new_accept) .+ V .* new_accept
        accepted = accepted .+ new_accept
    end

    gamma_std = d .* V_out

    u_boost = rand(rng, size(gamma_std)...)
    boost = ifelse.(boost_mask_f .> 0.5, 1.0, (1.0 .- u_boost) .^ (1.0 ./ alpha_orig))
    gamma_std = gamma_std .* boost

    return gamma_std ./ rate
end

sampler(::Type{<:Gamma}) = _gamma_sampler
sampler_traced(::Type{<:Gamma}) = true
support(::Type{<:Gamma}) = :positive
dist_spec(::Type{<:Gamma}) = _dist_spec("GAMMA")

#==============================================================================#
# StudentT Distribution
#==============================================================================#

struct StudentT{Tdf,Tloc,Tscale,S<:Tuple} <: AbstractDistribution
    df::Tdf
    loc::Tloc
    scale::Tscale
    shape::S

    function StudentT{Tdf,Tloc,Tscale,S}(
        df::Tdf, loc::Tloc, scale::Tscale, shape::S
    ) where {Tdf,Tloc,Tscale,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tdf,Tloc,Tscale,S}(df, loc, scale, shape)
    end
end

function StudentT(
    df::Tdf, loc::Tloc, scale::Tscale, shape::S
) where {Tdf,Tloc,Tscale,S<:Tuple}
    return StudentT{Tdf,Tloc,Tscale,S}(df, loc, scale, shape)
end
StudentT(df, loc, scale) = StudentT(df, loc, scale, (1,))

params(d::StudentT) = (d.df, d.loc, d.scale, d.shape)

function _studentt_sampler(rng, df, loc, scale, shape)
    z = randn(rng, shape)
    df_shaped = Ops.fill(0.0, shape...) .+ df
    chi2 = _gamma_sampler(rng, df_shaped ./ 2.0, 0.5, shape)
    y = z .* sqrt.(df_shaped ./ chi2)
    return loc .+ scale .* y
end

sampler(::Type{<:StudentT}) = _studentt_sampler
sampler_traced(::Type{<:StudentT}) = true
support(::Type{<:StudentT}) = :real
dist_spec(::Type{<:StudentT}) = _dist_spec("STUDENT_T")

#==============================================================================#
# HalfCauchy Distribution
#==============================================================================#

struct HalfCauchy{Tscale,S<:Tuple} <: AbstractDistribution
    scale::Tscale
    shape::S

    function HalfCauchy{Tscale,S}(scale::Tscale, shape::S) where {Tscale,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tscale,S}(scale, shape)
    end
end

function HalfCauchy(scale::Tscale, shape::S) where {Tscale,S<:Tuple}
    return HalfCauchy{Tscale,S}(scale, shape)
end
HalfCauchy(scale) = HalfCauchy(scale, (1,))

params(d::HalfCauchy) = (d.scale, d.shape)
support(::Type{<:HalfCauchy}) = :positive
dist_spec(::Type{<:HalfCauchy}) = _dist_spec("HALF_CAUCHY")

#==============================================================================#
# HalfStudentT Distribution
#==============================================================================#

struct HalfStudentT{Tdf,Tscale,S<:Tuple} <: AbstractDistribution
    df::Tdf
    scale::Tscale
    shape::S

    function HalfStudentT{Tdf,Tscale,S}(
        df::Tdf, scale::Tscale, shape::S
    ) where {Tdf,Tscale,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tdf,Tscale,S}(df, scale, shape)
    end
end

function HalfStudentT(df::Tdf, scale::Tscale, shape::S) where {Tdf,Tscale,S<:Tuple}
    return HalfStudentT{Tdf,Tscale,S}(df, scale, shape)
end
HalfStudentT(df, scale) = HalfStudentT(df, scale, (1,))

params(d::HalfStudentT) = (d.df, d.scale, d.shape)

function _halfstudentt_sampler(rng, df, scale, shape)
    return abs.(_studentt_sampler(rng, df, 0.0, scale, shape))
end

sampler(::Type{<:HalfStudentT}) = _halfstudentt_sampler
sampler_traced(::Type{<:HalfStudentT}) = true
support(::Type{<:HalfStudentT}) = :positive
dist_spec(::Type{<:HalfStudentT}) = _dist_spec("HALF_STUDENT_T")

#==============================================================================#
# Dirichlet Distribution (simplex along the last axis)
#==============================================================================#

struct Dirichlet{Tα,S<:Tuple} <: AbstractDistribution
    alpha::Tα
    shape::S

    function Dirichlet{Tα,S}(alpha::Tα, shape::S) where {Tα,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tα,S}(alpha, shape)
    end
end

Dirichlet(alpha::Tα, shape::S) where {Tα,S<:Tuple} = Dirichlet{Tα,S}(alpha, shape)
Dirichlet(alpha) = Dirichlet(alpha, (length(alpha),))

params(d::Dirichlet) = (d.alpha, d.shape)

function _dirichlet_sampler(rng, alpha, shape)
    # Not bitwise equal to jax.random.dirichlet (softmax of loggamma).
    g = _gamma_sampler(rng, alpha, 1.0, shape)
    return g ./ sum(g; dims=ndims(g))
end

sampler(::Type{<:Dirichlet}) = _dirichlet_sampler
sampler_traced(::Type{<:Dirichlet}) = true
support(::Type{<:Dirichlet}) = :simplex
dist_spec(::Type{<:Dirichlet}) = _dist_spec("DIRICHLET")

#==============================================================================#
# DiscreteUniform Distribution (inclusive bounds)
#==============================================================================#

struct DiscreteUniform{Tlow,Thigh,S<:Tuple} <: AbstractDistribution
    low::Tlow
    high::Thigh
    shape::S

    function DiscreteUniform{Tlow,Thigh,S}(
        low::Tlow, high::Thigh, shape::S
    ) where {Tlow,Thigh,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tlow,Thigh,S}(low, high, shape)
    end
end

function DiscreteUniform(low::Tlow, high::Thigh, shape::S) where {Tlow,Thigh,S<:Tuple}
    return DiscreteUniform{Tlow,Thigh,S}(low, high, shape)
end
DiscreteUniform(low, high) = DiscreteUniform(low, high, (1,))

params(d::DiscreteUniform) = (d.low, d.high, d.shape)
support(::Type{<:DiscreteUniform}) = :real
dist_spec(::Type{<:DiscreteUniform}) = _dist_spec("DISCRETE_UNIFORM")

#==============================================================================#
# Gumbel Distribution
#==============================================================================#

struct Gumbel{Tloc,Tscale,S<:Tuple} <: AbstractDistribution
    loc::Tloc
    scale::Tscale
    shape::S

    function Gumbel{Tloc,Tscale,S}(
        loc::Tloc, scale::Tscale, shape::S
    ) where {Tloc,Tscale,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tloc,Tscale,S}(loc, scale, shape)
    end
end

function Gumbel(loc::Tloc, scale::Tscale, shape::S) where {Tloc,Tscale,S<:Tuple}
    return Gumbel{Tloc,Tscale,S}(loc, scale, shape)
end
Gumbel(loc, scale) = Gumbel(loc, scale, (1,))
Gumbel() = Gumbel(0.0, 1.0, (1,))

params(d::Gumbel) = (d.loc, d.scale, d.shape)
support(::Type{<:Gumbel}) = :real
dist_spec(::Type{<:Gumbel}) = _dist_spec("GUMBEL")
