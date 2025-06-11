using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function generate_model(seed, μ, σ, shape)
    function model(seed, μ, σ, shape)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        s = ProbProg.sample!(normal, rng, μ, σ, shape)
        t = ProbProg.sample!(normal, rng, s, σ, shape)
        return t
    end

    return ProbProg.generate!(model, seed, μ, σ, shape)
end

@testset "Generate" begin
    @testset "normal_deterministic" begin
        shape = (10000,)
        seed1 = Reactant.to_rarray(UInt64[1, 4])
        seed2 = Reactant.to_rarray(UInt64[1, 4])
        μ1 = Reactant.ConcreteRNumber(0.0)
        μ2 = Reactant.ConcreteRNumber(1000.0)
        σ1 = Reactant.ConcreteRNumber(1.0)
        σ2 = Reactant.ConcreteRNumber(1.0)

        model_compiled = @compile optimize = :probprog generate_model(seed1, μ1, σ1, shape)

        @test Array(model_compiled(seed1, μ1, σ1, shape)) ≈
            Array(model_compiled(seed1, μ1, σ1, shape))
        @test mean(Array(model_compiled(seed1, μ1, σ1, shape))) ≈ 0.0 atol = 0.05 rtol =
            0.05
        @test mean(Array(model_compiled(seed2, μ2, σ2, shape))) ≈ 1000.0 atol = 0.05 rtol =
            0.05
        @test !(all(
            Array(model_compiled(seed1, μ1, σ1, shape)) .≈
            Array(model_compiled(seed2, μ2, σ2, shape)),
        ))
    end
    @testset "normal_hlo" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = :no_enzyme generate_model(seed, μ, σ, shape)
        @test contains(repr(before), "enzyme.generate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog generate_model(seed, μ, σ, shape)
        @test !contains(repr(after), "enzyme.generate")
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "normal_generate" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        X = Array(@jit optimize = :probprog generate_model(seed, μ, σ, shape))
        @test mean(X) ≈ 0.0 atol = 0.05 rtol = 0.05
    end
end
