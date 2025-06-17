using Reactant, Test, Random
using Reactant: ProbProg

function normal(rng, μ, σ, shape)
    return μ .+ σ .* randn(rng, shape)
end

function bernoulli_logit(rng, logit, shape)
    return rand(rng, shape...) .< (1 ./ (1 .+ exp.(-logit)))
end

function blr(seed, N, K)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # α ~ Normal(0, 10, size = 1)
    α = ProbProg.sample!(normal, rng, 0, 10, (1,); symbol=:α)

    # β ~ Normal(0, 2.5, size = K)
    β = ProbProg.sample!(normal, rng, 0, 2.5, (K,); symbol=:β)

    # X ~ Normal(0, 10, size = (N, K))
    X = ProbProg.sample!(normal, rng, 0, 10, (N, K); symbol=:X)

    # μ = α .+ X * β
    μ = α .+ X * β

    Y = ProbProg.sample!(bernoulli_logit, rng, μ, (N,); symbol=:Y)

    return Y
end

@testset "BLR" begin
    N = 5  # number of observations
    K = 3  # number of features
    seed = Reactant.to_rarray(UInt64[1, 4])

    trace = ProbProg.create_trace()

    @test size(
        Array(@jit optimize = :probprog ProbProg.simulate!(blr, seed, N, K; trace))
    ) == (N,)

    ProbProg.print_trace(trace)
end
