using Random: randexp
using LinearAlgebra: cholesky, I

abstract type Distribution end

function sampler(::Type{D}) where {D<:Distribution}
    return error("sampler not implemented for $D")
end

# TODO(#2542): logpdf_fn should return a vector of log probabilities for each element of the input
function logpdf_fn(::Type{D}) where {D<:Distribution}
    return error("logpdf_fn not implemented for $D")
end

function params(d::Distribution)
    return error("params not implemented for $(typeof(d))")
end

function support(::Type{D}) where {D<:Distribution}
    return error("support not implemented for $D")
end

function bounds(::Type{D}) where {D<:Distribution}
    return (nothing, nothing)
end

# F2 spike: the `#impulse.dist` attribute string for distributions the shared
# builder can emit end-to-end, or `nothing` for distributions still carried by
# traced sampler/logpdf symbols. Only consulted when `EMIT_DISTRIBUTIONS[]`.
function dist_spec(::Type{D}) where {D<:Distribution}
    return nothing
end

#==============================================================================#
# Normal Distribution
#==============================================================================#

struct Normal{Tμ,Tσ,S<:Tuple} <: Distribution
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

function _normal_sampler(rng, μ, σ, shape)
    return μ .+ σ .* randn(rng, shape)
end

# Reference: https://github.com/probcomp/Gen.jl/blob/027dbc6e71cec3e281b335292c57ad776a8bf9db/src/modeling_library/distributions/normal.jl#L61-L69
function _normal_logpdf(x, μ, σ, _)
    z = (x .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ))
end

sampler(::Type{<:Normal}) = _normal_sampler
logpdf_fn(::Type{<:Normal}) = _normal_logpdf
support(::Type{<:Normal}) = :real
# Normal(mu, sigma) = StdNormal pushed through Affine(loc=mu, scale=sigma).
dist_spec(::Type{<:Normal}) = "#impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>"

#==============================================================================#
# Exponential Distribution
#==============================================================================#

struct Exponential{Tλ,S<:Tuple} <: Distribution
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

function _exponential_sampler(rng, λ, shape)
    exp1_sample = randexp(rng, shape)
    return exp1_sample ./ λ
end

function _exponential_logpdf(x, λ, _)
    return sum(log.(λ) .- λ .* x)
end

sampler(::Type{<:Exponential}) = _exponential_sampler
logpdf_fn(::Type{<:Exponential}) = _exponential_logpdf
support(::Type{<:Exponential}) = :positive
# Native Exponential(lambda): sampler -log1p(-u)/lambda, split logpdf.
dist_spec(::Type{<:Exponential}) = "#impulse.dist<EXPONENTIAL>"
#==============================================================================#
# HalfNormal Distribution
#==============================================================================#

struct HalfNormal{Tσ,S<:Tuple} <: Distribution
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

function _halfnormal_sampler(rng, σ, shape)
    return abs.(σ .* randn(rng, shape))
end

function _halfnormal_logpdf(x, σ, _)
    n = length(x)
    log_sigma = sum(log.(σ))
    return n * log(2) - n * log_sigma - n / 2 * log(2π) - sum(x .* x ./ (2 .* σ .* σ))
end

sampler(::Type{<:HalfNormal}) = _halfnormal_sampler
logpdf_fn(::Type{<:HalfNormal}) = _halfnormal_logpdf
support(::Type{<:HalfNormal}) = :positive
# Folded native HalfNormal(sigma): sampler abs(sigma*z), split logpdf. Not a
# base+chain because abs is not a bijection.
dist_spec(::Type{<:HalfNormal}) = "#impulse.dist<HALF_NORMAL>"

#==============================================================================#
# LogNormal Distribution
#==============================================================================#

struct LogNormal{Tμ,Tσ,S<:Tuple} <: Distribution
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

function _lognormal_sampler(rng, μ, σ, shape)
    z = μ .+ σ .* randn(rng, shape)
    return exp.(z)
end

function _lognormal_logpdf(x, μ, σ, _)
    log_x = log.(x)
    z = (log_x .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ) .- log_x)
end

sampler(::Type{<:LogNormal}) = _lognormal_sampler
logpdf_fn(::Type{<:LogNormal}) = _lognormal_logpdf
support(::Type{<:LogNormal}) = :positive
# LogNormal(mu, sigma) = StdNormal pushed through Affine(loc=mu, scale=sigma) then
# Exp; sampler exp(mu + sigma*z), logpdf base_logpdf(inverse(y)) - Sum logdet
# (matches the traced -log(sigma) - log(x) change-of-variables).
function dist_spec(::Type{<:LogNormal})
    return "#impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>"
end

struct Bernoulli{Tp,S<:Tuple} <: Distribution
    logits::Tp
    shape::S

    function Bernoulli{Tp,S}(logits::Tp, shape::S) where {Tp,S<:Tuple}
        isempty(shape) && throw(ArgumentError("shape cannot be empty"))
        return new{Tp,S}(logits, shape)
    end
end

Bernoulli(logits::Tp, shape::S) where {Tp,S<:Tuple} = Bernoulli{Tp,S}(logits, shape)

params(d::Bernoulli) = (d.logits, d.shape)

function _bernoulli_sampler(rng, logits, shape)
    probs = 1.0 ./ (1.0 .+ exp.(-logits))
    u = rand(rng, size(probs)...)
    return Float64.(u .< probs)
end

function _bernoulli_logpdf(y, logits, _)
    return sum(y .* logits .- max.(logits, 0.0) .- log1p.(exp.(.-abs.(logits))))
end

sampler(::Type{<:Bernoulli}) = _bernoulli_sampler
logpdf_fn(::Type{<:Bernoulli}) = _bernoulli_logpdf
support(::Type{<:Bernoulli}) = :real
# Native Bernoulli(logits): sampler u < sigmoid(logits), split logpdf.
dist_spec(::Type{<:Bernoulli}) = "#impulse.dist<BERNOULLI>"
#==============================================================================#
# Poisson Distribution
#==============================================================================#

struct Poisson{Tλ,S<:Tuple} <: Distribution
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
# Iteratively samples uniform(0,1), accumulates log-products, and counts
# arrivals until the product drops below exp(-λ).
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
# Uses precomputed envelope parameters for efficient rejection sampling.
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

        # accept1: fast path (u_shifted >= 0.07 AND v <= v_r)
        accept1_f = ifelse.(u_shifted .>= 0.07, 1.0, 0.0) .* ifelse.(v .<= v_r, 1.0, 0.0)
        # reject: k < 0 OR (u_shifted < 0.013 AND v > u_shifted)
        reject_f =
            min.(
                ifelse.(k .< 0.0, 1.0, 0.0) .+
                ifelse.(u_shifted .< 0.013, 1.0, 0.0) .* ifelse.(v .> u_shifted, 1.0, 0.0),
                1.0,
            )
        # accept2: full check (s <= t)
        accept2_f = ifelse.(s .<= t, 1.0, 0.0)
        # accept: accept1 OR (NOT reject AND accept2)
        accept = min.(accept1_f .+ (1.0 .- reject_f) .* accept2_f, 1.0)

        # Only update elements not yet accepted
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

    # Run both algorithms with masked rates and select result.
    # Knuth elements get rate=0 for rejection (instant termination),
    # rejection elements get rate=1e5 for Knuth (valid but discarded).
    lam_knuth = rate_shaped .* use_knuth_f
    lam_rejection = rate_shaped .* (1.0 .- use_knuth_f) .+ 1e5 .* use_knuth_f

    k_knuth = _poisson_knuth(rng, lam_knuth, shape)
    k_rejection = _poisson_rejection(rng, lam_rejection, shape)

    result = k_knuth .* use_knuth_f .+ k_rejection .* (1.0 .- use_knuth_f)

    # Handle rate == 0
    nonzero_f = ifelse.(rate_shaped .> 0.0, 1.0, 0.0)
    return result .* nonzero_f
end

function _poisson_logpdf(k, rate, _)
    return sum(k .* log.(rate) .- rate .- Ops.lgamma(k .+ 1.0))
end

sampler(::Type{<:Poisson}) = _poisson_sampler
logpdf_fn(::Type{<:Poisson}) = _poisson_logpdf
support(::Type{<:Poisson}) = :real
# Native Poisson(rate): logpdf emitted (needs impulse.lgamma); sampler stays
# traced (Knuth / Hoermann rejection loops are not bitwise-reproducible).
dist_spec(::Type{<:Poisson}) = "#impulse.dist<POISSON>"
#==============================================================================#
# Multivariate Normal Distribution
#==============================================================================#

struct MultivariateNormal{Tμ,TΣ,S<:Tuple} <: Distribution
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

function _mvn_sampler(rng, μ, Σ, shape)
    n = shape[1]
    z = randn(rng, shape)
    C = cholesky(Σ)
    return μ .+ reshape(C.factors' * reshape(z, n, 1), n)
end

function _mvn_logpdf(x, μ, Σ, shape)
    n = shape[1]
    C = cholesky(Σ)
    diff = x .- μ
    alpha = reshape(C \ reshape(diff, n, 1), n)
    I_n = Matrix{Float64}(I, n, n)
    U_diag = vec(sum(C.factors .* I_n; dims=1))
    log_det = 2.0 * sum(log.(U_diag))
    return -0.5 * (sum(diff .* alpha) + log_det + n * log(2π))
end

sampler(::Type{<:MultivariateNormal}) = _mvn_sampler
logpdf_fn(::Type{<:MultivariateNormal}) = _mvn_logpdf
support(::Type{<:MultivariateNormal}) = :real
#==============================================================================#
# Gamma Distribution
# Parameterized by concentration (α) and rate (β), matching NumPyro's convention.
# NumPyro: Gamma(concentration, rate) with rate = 1/scale
#==============================================================================#

struct Gamma{Tα,Tβ,S<:Tuple} <: Distribution
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
    # Reference: "A Simple Method for Generating Gamma Variables" - Marsaglia & Tsang
    alpha = Ops.fill(0.0, shape...) .+ concentration

    # For alpha < 1, boost to alpha + 1: Gamma(α) = Gamma(α+1) * U^(1/α)
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

        # Squeeze test: u < 1 - 0.0331 * X^2
        squeeze_f = ifelse.(u .< 1.0 .- 0.0331 .* X .* X, 1.0, 0.0)
        # Log test: log(u) < X/2 + d*(1 - V + log(V))
        log_f = ifelse.(log.(u) .< X ./ 2.0 .+ d .* (1.0 .- V .+ log.(V)), 1.0, 0.0)

        accept = v_pos_f .* min.(squeeze_f .+ log_f, 1.0)

        not_yet = 1.0 .- min.(accepted, 1.0)
        new_accept = accept .* not_yet
        V_out = V_out .* (1.0 .- new_accept) .+ V .* new_accept
        accepted = accepted .+ new_accept
    end

    gamma_std = d .* V_out

    # Boost correction for alpha < 1: Gamma(α) = Gamma(α+1) * (1-U)^(1/α)
    u_boost = rand(rng, size(gamma_std)...)
    boost = ifelse.(boost_mask_f .> 0.5, 1.0, (1.0 .- u_boost) .^ (1.0 ./ alpha_orig))
    gamma_std = gamma_std .* boost

    return gamma_std ./ rate
end

function _gamma_logpdf(x, concentration, rate, _)
    # Matches NumPyro: (α-1)*log(x) - β*x - lgamma(α) + α*log(β)
    # Broadcast params to x's shape to ensure they are traced for Ops.lgamma
    α = Ops.fill(0.0, size(x)...) .+ concentration
    β = Ops.fill(0.0, size(x)...) .+ rate
    return sum((α .- 1) .* log.(x) .- β .* x .- Ops.lgamma(α) .+ α .* log.(β))
end

sampler(::Type{<:Gamma}) = _gamma_sampler
logpdf_fn(::Type{<:Gamma}) = _gamma_logpdf
support(::Type{<:Gamma}) = :positive
# Native Gamma(concentration, rate): logpdf emitted (needs impulse.lgamma);
# sampler stays traced (Marsaglia-Tsang rejection loop).
dist_spec(::Type{<:Gamma}) = "#impulse.dist<GAMMA>"
#==============================================================================#
# StudentT Distribution
# Parameterized by df (degrees of freedom), loc, scale.
# Matches NumPyro's StudentT(df, loc, scale).
#==============================================================================#

struct StudentT{Tdf,Tloc,Tscale,S<:Tuple} <: Distribution
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
    # Matches NumPyro: loc + scale * Normal(0,1) / sqrt(Chi2(df) / df)
    # where Chi2(df) = Gamma(df/2, rate=0.5)
    z = randn(rng, shape)
    df_shaped = Ops.fill(0.0, shape...) .+ df
    chi2 = _gamma_sampler(rng, df_shaped ./ 2.0, 0.5, shape)
    y = z .* sqrt.(df_shaped ./ chi2)
    return loc .+ scale .* y
end

function _studentt_logpdf(x, df, loc, scale, _)
    # Matches NumPyro:
    # lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*π) - log(scale)
    # - (df+1)/2 * log1p(((x-loc)/scale)^2 / df)
    # Broadcast params to x's shape to ensure they are traced for Ops.lgamma
    df_t = Ops.fill(0.0, size(x)...) .+ df
    scale_t = Ops.fill(0.0, size(x)...) .+ scale
    y = (x .- loc) ./ scale_t
    return sum(
        Ops.lgamma((df_t .+ 1) ./ 2) .- Ops.lgamma(df_t ./ 2) .- 0.5 .* log.(df_t .* π) .-
        log.(scale_t) .- (df_t .+ 1) ./ 2 .* log1p.(y .* y ./ df_t),
    )
end

sampler(::Type{<:StudentT}) = _studentt_sampler
logpdf_fn(::Type{<:StudentT}) = _studentt_logpdf
support(::Type{<:StudentT}) = :real
# Native StudentT(df, loc, scale): logpdf emitted (needs impulse.lgamma);
# sampler stays traced (draws a Gamma via rejection).
dist_spec(::Type{<:StudentT}) = "#impulse.dist<STUDENT_T>"
#==============================================================================#
# HalfCauchy Distribution
# Parameterized by scale. Support: [0, ∞).
# Matches NumPyro's HalfCauchy(scale).
#==============================================================================#

struct HalfCauchy{Tscale,S<:Tuple} <: Distribution
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

function _halfcauchy_sampler(rng, scale, shape)
    # Matches JAX: abs(scale * cauchy_standard) where
    # cauchy_standard = tan(π * (u - 0.5)), u ~ Uniform(eps, 1)
    u = rand(rng, shape...)
    finfo_eps = eps(Float64)
    u_cauchy = u .* (1.0 - finfo_eps) .+ finfo_eps
    return scale .* abs.(tan.(π .* (u_cauchy .- 0.5)))
end

function _halfcauchy_logpdf(x, scale, _)
    # Matches NumPyro: log(2/π) - log(scale) - log(1 + (x/scale)^2)
    return sum(log(2 / π) .- log.(scale) .- log1p.((x ./ scale) .* (x ./ scale)))
end

sampler(::Type{<:HalfCauchy}) = _halfcauchy_sampler
logpdf_fn(::Type{<:HalfCauchy}) = _halfcauchy_logpdf
support(::Type{<:HalfCauchy}) = :positive
# Folded native HalfCauchy(scale): sampler scale*abs(tan(pi*(u_cauchy-0.5))),
# split logpdf. Not a base+chain because abs is not a bijection.
dist_spec(::Type{<:HalfCauchy}) = "#impulse.dist<HALF_CAUCHY>"
#==============================================================================#
# Dirichlet Distribution
# Parameterized by a concentration vector `alpha` over a K-simplex (shape (K,)).
# Support: SIMPLEX (stick-breaking). Matches scipy.stats.dirichlet.
#==============================================================================#

struct Dirichlet{Tα,S<:Tuple} <: Distribution
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
    # Gamma-normalize construction. NOTE: jax.random.dirichlet uses
    # `softmax(loggamma(key, alpha))` (a log-space rejection sampler), so this is
    # only distributionally equal, not bit-identical (different RNG consumption).
    # For a batched (m, K) site the K-simplex event axis is the last axis, so the
    # normalization is per K-block, not global.
    g = _gamma_sampler(rng, alpha, 1.0, shape)
    return g ./ sum(g; dims=ndims(g))
end

function _dirichlet_logpdf(x, alpha, _)
    # Term-for-term match of numpyro.distributions.Dirichlet.log_prob, reducing
    # over the last (event) axis so that a batched (m, K) site is m independent
    # K-simplices:
    #   normalize_term = sum(gammaln(conc), -1) - gammaln(sum(conc, -1))
    #   log_prob       = sum(log(value) * (conc - 1), -1) - normalize_term
    # The scalar returned is the sum of the per-block log_prob values.
    α = Ops.fill(0.0, size(x)...) .+ alpha
    d = ndims(α)
    normalize_term = sum(Ops.lgamma(α); dims=d) .- Ops.lgamma(sum(α; dims=d))
    return sum(sum(log.(x) .* (α .- 1.0); dims=d) .- normalize_term)
end

sampler(::Type{<:Dirichlet}) = _dirichlet_sampler
logpdf_fn(::Type{<:Dirichlet}) = _dirichlet_logpdf
support(::Type{<:Dirichlet}) = :simplex
# Distributional native Dirichlet(alpha): logpdf emitted with per-block (last
# axis) reduction (needs impulse.lgamma); the gamma-normalize sampler stays
# traced (distributional, not bitwise, per B4).
dist_spec(::Type{<:Dirichlet}) = "#impulse.dist<DIRICHLET>"
#==============================================================================#
# DiscreteUniform Distribution
# Uniform over the integers {low, low+1, ..., high} (both endpoints included).
#==============================================================================#

struct DiscreteUniform{Tlow,Thigh,S<:Tuple} <: Distribution
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

function _discreteuniform_sampler(rng, low, high, shape)
    # numpyro draws `random.randint(key, shape, low, high + 1)`; jax's randint
    # uses a two-word rejection-free algorithm we cannot reproduce bit-for-bit
    # with a single uniform draw, so this is distributionally equal but not
    # bitwise identical to numpyro's sampler.
    n = (high - low) + 1.0
    u = rand(rng, shape...)
    return low .+ floor.(u .* n)
end

function _discreteuniform_logpdf(x, low, high, _)
    # numpyro: -log(high + 1 - low), constant over the support, broadcast to the
    # sample shape and summed.
    logp = .-log.((high + 1.0) - low)
    return sum(logp .+ zero.(x))
end

sampler(::Type{<:DiscreteUniform}) = _discreteuniform_sampler
logpdf_fn(::Type{<:DiscreteUniform}) = _discreteuniform_logpdf
support(::Type{<:DiscreteUniform}) = :real
# Native DiscreteUniform(low, high): sampler low + floor(u*(high-low+1)), split
# logpdf. low/high must be passed as traced operands (not literals) for emit.
dist_spec(::Type{<:DiscreteUniform}) = "#impulse.dist<DISCRETE_UNIFORM>"
