using Reactant, Test, Random, Statistics
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
normal_logpdf(x, μ, σ, _) = -sum(log.(σ)) - sum((μ .- x) .^ 2) / (2 * σ^2)

function model(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(normal, rng, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    t = ProbProg.sample(normal, rng, s, σ, shape; symbol=:t, logpdf=normal_logpdf)
    return t
end

@testset "Generate" begin
    @testset "unconstrained" begin
        shape = (1000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        trace, weight = ProbProg.generate(model, seed, μ, σ, shape)
        @test mean(trace.retval) ≈ 0.0 atol = 0.05 rtol = 0.05
    end

    @testset "constrained" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint = Dict{Symbol,Any}(:s => (fill(0.1, shape),))

        trace, weight = ProbProg.generate(model, seed, μ, σ, shape; constraint)

        @test trace.choices[:s] == constraint[:s]

        expected_weight =
            normal_logpdf(constraint[:s][1], 0.0, 1.0, shape) +
            normal_logpdf(trace.choices[:t][1], constraint[:s][1], 1.0, shape)
        @test weight ≈ expected_weight atol = 1e-6
    end
end
