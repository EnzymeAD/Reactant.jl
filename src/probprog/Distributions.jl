abstract type Distribution end

sampler(::Type{D}) where {D<:Distribution} = error("sampler not implemented for $D")
logpdf_fn(::Type{D}) where {D<:Distribution} = error("logpdf_fn not implemented for $D")
params(d::Distribution) = error("params not implemented for $(typeof(d))")

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
    z = (x .- μ) ./ σ
    n = length(x)
    return -n * log(σ) - n / 2 * log(2π) - sum(z .^ 2) / 2
end

sampler(::Type{<:Normal}) = _normal_sampler
logpdf_fn(::Type{<:Normal}) = _normal_logpdf
