using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

normal(rng, μ, σ) = μ .+ σ .* randn(rng, 10000)

function generate_model(seed, μ, σ)
    function model(seed, μ, σ)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        s = ProbProg.sample(normal, rng, μ, σ)
        t = ProbProg.sample(normal, rng, s, σ)
        return t
    end

    return ProbProg.generate(model, seed, μ, σ)
end

@testset "ProbProg" begin
    @testset "normal_deterministic" begin
        seed1 = Reactant.to_rarray(UInt64[1, 4])
        seed2 = Reactant.to_rarray(UInt64[1, 4])
        μ1 = Reactant.ConcreteRArray(0.0)
        μ2 = Reactant.ConcreteRArray(1000.0)
        σ1 = Reactant.ConcreteRArray(1.0)
        σ2 = Reactant.ConcreteRArray(1.0)
        model_compiled = @compile generate_model(seed1, μ1, σ1)

        @test Array(model_compiled(seed1, μ1, σ1)) ≈ Array(model_compiled(seed1, μ1, σ1))
        @test mean(Array(model_compiled(seed1, μ1, σ1))) ≈ 0.0 atol = 0.05 rtol = 0.05
        @test mean(Array(model_compiled(seed2, μ2, σ2))) ≈ 1000.0 atol = 0.05 rtol = 0.05
        @test !(all(
            Array(model_compiled(seed1, μ1, σ1)) .≈ Array(model_compiled(seed2, μ2, σ2))
        ))
    end
    @testset "normal_hlo" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRArray(0.0)
        σ = Reactant.ConcreteRArray(1.0)
        before = @code_hlo optimize = :none generate_model(seed, μ, σ)
        @test contains(repr(before), "enzyme.generate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog generate_model(seed, μ, σ)
        @test !contains(repr(after), "enzyme.generate")
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "normal_generate" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRArray(0.0)
        σ = Reactant.ConcreteRArray(1.0)
        X = Array(@jit optimize = :probprog generate_model(seed, μ, σ))
        @test mean(X) ≈ 0.0 atol = 0.05 rtol = 0.05
    end
end
