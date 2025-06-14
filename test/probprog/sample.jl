using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

@noinline normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function sample1(seed, μ, σ, shape)
    function model(seed, μ, σ, shape)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        s = ProbProg.sample!(normal, rng, μ, σ, shape)
        return s
    end

    return ProbProg.generate!(model, seed, μ, σ, shape)
end

function sample2(seed, μ, σ, shape)
    function model(seed, μ, σ, shape)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        s = ProbProg.sample!(normal, rng, μ, σ, shape)
        t = ProbProg.sample!(normal, rng, μ, σ, shape)
        return t
    end

    return ProbProg.generate!(model, seed, μ, σ, shape)
end

@testset "test" begin
    @testset "sample_hlo" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        before = @code_hlo optimize = false sample2(seed, μ, σ, shape)
        @test contains(repr(before), "enzyme.sample")
        after = @code_hlo optimize = :probprog sample2(seed, μ, σ, shape)
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "sample_normal" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        X = Array(@jit optimize = :probprog sample1(seed, μ, σ, shape))
        Y = Array(@jit optimize = :probprog sample2(seed, μ, σ, shape))
        @test !all(X .≈ Y)
    end
end
