using Random: randexp

abstract type Distribution end

function sampler(::Type{D}) where {D<:Distribution}
    return error("sampler not implemented for $D")
end

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

function _normal_logpdf(x, μ, σ, _)
    n = length(x)
    log_sigma = sum(log.(σ))
    diff = x .- μ
    return -n * log_sigma - n / 2 * log(2π) - sum(diff .* diff ./ (2 .* σ .* σ))
end

sampler(::Type{<:Normal}) = _normal_sampler
logpdf_fn(::Type{<:Normal}) = _normal_logpdf
support(::Type{<:Normal}) = :real

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
    n = length(x)
    log_lambda = sum(log.(λ))
    return n * log_lambda - sum(λ .* x)
end

sampler(::Type{<:Exponential}) = _exponential_sampler
logpdf_fn(::Type{<:Exponential}) = _exponential_logpdf
support(::Type{<:Exponential}) = :positive

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
    n = length(x)
    log_x = log.(x)
    log_sigma = sum(log.(σ))
    diff = log_x .- μ
    return -sum(log_x) - n * log_sigma - n / 2 * log(2π) -
           sum(diff .* diff ./ (2 .* σ .* σ))
end

sampler(::Type{<:LogNormal}) = _lognormal_sampler
logpdf_fn(::Type{<:LogNormal}) = _lognormal_logpdf
support(::Type{<:LogNormal}) = :positive
