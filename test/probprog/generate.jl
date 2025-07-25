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
    @testset "hlo" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = :no_enzyme ProbProg.generate_internal(
            model, seed, μ, σ, shape; trace=ProbProg.ProbProgTrace()
        )
        @test contains(repr(before), "enzyme.generate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog ProbProg.generate_internal(
            model, seed, μ, σ, shape; trace=ProbProg.ProbProgTrace()
        )
        @test !contains(repr(after), "enzyme.generate")
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "normal" begin
        shape = (1000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        trace, weight = ProbProg.generate(model, seed, μ, σ, shape)
        @test mean(trace.retval) ≈ 0.0 atol = 0.05 rtol = 0.05
    end

    @testset "constraints" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        s_constraint = fill(0.1, shape)
        constraints = Dict(:s => [s_constraint])

        trace, weight = ProbProg.generate(model, seed, μ, σ, shape; constraints)

        @test trace.choices[:s] == s_constraint

        expected_weight =
            normal_logpdf(s_constraint, 0.0, 1.0, shape) +
            normal_logpdf(trace.choices[:t], s_constraint, 1.0, shape)
        @test weight ≈ expected_weight atol = 1e-6
    end
end
