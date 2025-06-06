using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg
using Libdl: Libdl

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
bernoulli_logit(rng, logit, shape) = rand(rng, shape...) .< (1 ./ (1 .+ exp.(-logit)))

function blr(seed, N, K)
    function model(seed, N, K)
        rng = Random.default_rng()
        Random.seed!(rng, seed)

        # α ~ Normal(0, 10, size = 1)
        α = ProbProg.sample!(normal, rng, 0, 10, (1,); symbol=:α)

        # β ~ Normal(0, 2.5, size = K)
        β = ProbProg.sample!(normal, rng, 0, 2.5, (K,); symbol=:β)

        # X ~ Normal(0, 10, size = (N, K))
        X = ProbProg.sample!(normal, rng, 0, 10, (N, K); symbol=:X) # TODO: double check transpose 

        # μ = α .+ X * β
        μ = α .+ X * β

        ProbProg.sample!(bernoulli_logit, rng, μ, (N,); symbol=:Y)

        return μ
    end

    return ProbProg.simulate!(model, seed, N, K)
end

@testset "BLR" begin
    N = 5  # number of observations
    K = 3  # number of features
    seed = Reactant.to_rarray(UInt64[1, 4])

    X = ProbProg.getTrace(@jit optimize = :probprog blr(seed, N, K))
    ProbProg.print_trace(X)
end
