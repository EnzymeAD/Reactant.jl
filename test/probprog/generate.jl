using Reactant, Test, Random, Statistics
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function model(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(normal, rng, μ, σ, shape)
    t = ProbProg.sample(normal, rng, s, σ, shape)
    return t
end

@testset "Generate" begin
    @testset "deterministic" begin
        shape = (10000,)
        seed1 = Reactant.to_rarray(UInt64[1, 4])
        seed2 = Reactant.to_rarray(UInt64[1, 4])
        μ1 = Reactant.ConcreteRNumber(0.0)
        μ2 = Reactant.ConcreteRNumber(1000.0)
        σ1 = Reactant.ConcreteRNumber(1.0)
        σ2 = Reactant.ConcreteRNumber(1.0)

        generate_model(seed, μ, σ, shape) =
            ProbProg.generate_internal(model, seed, μ, σ, shape)

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
    @testset "hlo" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = :no_enzyme ProbProg.generate_internal(
            model, seed, μ, σ, shape
        )
        @test contains(repr(before), "enzyme.generate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog ProbProg.generate_internal(
            model, seed, μ, σ, shape
        )
        @test !contains(repr(after), "enzyme.generate")
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "normal" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        X = ProbProg.generate(model, seed, μ, σ, shape)
        @test mean(X) ≈ 0.0 atol = 0.05 rtol = 0.05
    end

    @testset "correctness" begin
        op(x, y) = x * y'

        function fake_model(x, y)
            return ProbProg.sample(op, x, y)
        end

        x = reshape(collect(Float64, 1:12), (4, 3))
        y = reshape(collect(Float64, 1:12), (4, 3))
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test ProbProg.generate(fake_model, x_ra, y_ra) == op(x, y)
    end
end
