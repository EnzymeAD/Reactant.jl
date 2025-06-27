using Reactant, Test, Random
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function one_sample(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(normal, rng, μ, σ, shape)
    return s
end

function two_samples(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    _ = ProbProg.sample(normal, rng, μ, σ, shape)
    t = ProbProg.sample(normal, rng, μ, σ, shape)
    return t
end

@testset "test" begin
    @testset "sample_hlo" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        before = @code_hlo optimize = false ProbProg.call_internal(
            one_sample, seed, μ, σ, shape
        )
        @test contains(repr(before), "enzyme.sample")
        after = @code_hlo optimize = :probprog ProbProg.call_internal(
            two_samples, seed, μ, σ, shape
        )
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "rng_state" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        X = ProbProg.call(one_sample, seed, μ, σ, shape)
        Y = ProbProg.call(two_samples, seed, μ, σ, shape)
        @test !all(X .≈ Y)
    end
end
