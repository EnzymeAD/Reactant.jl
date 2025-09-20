using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -sum(log.(σ)) - length(x) / 2 * log(2π) - sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

bernoulli_logit(rng, logit, shape) = rand(rng, shape...) .< (1 ./ (1 .+ exp.(-logit)))
bernoulli_logit_logpdf(x, logit, _) = sum(x .* logit .- log1p.(exp.(logit)))

# https://github.com/facebookresearch/pplbench/blob/main/pplbench/models/logistic_regression.py
function blr(rng, N, K)
    # α ~ Normal(0, 10, size = 1)
    α = ProbProg.sample(rng, normal, 0, 10, (1,); symbol=:α, logpdf=normal_logpdf)

    # β ~ Normal(0, 2.5, size = K)
    β = ProbProg.sample(rng, normal, 0, 2.5, (K,); symbol=:β, logpdf=normal_logpdf)

    # X ~ Normal(0, 10, size = (N, K))
    X = ProbProg.sample(rng, normal, 0, 10, (N, K); symbol=:X, logpdf=normal_logpdf)

    # μ = α .+ X * β
    μ = α .+ X * β

    Y = ProbProg.sample(
        rng, bernoulli_logit, μ, (N,); symbol=:Y, logpdf=bernoulli_logit_logpdf
    )

    return Y
end

@testset "BLR" begin
    N = 5  # number of observations
    K = 3  # number of features
    seed = Reactant.to_rarray(UInt64[1, 4])

    rng = ReactantRNG(seed)

    trace, _ = ProbProg.simulate(rng, blr, N, K)

    @test size(trace.retval[1]) == (N,)
end
