using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

function one_sample(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
    return s
end

function two_samples(rng, μ, σ, shape)
    _ = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
    _, t = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
    return t
end

function compose(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
    _, t = ProbProg.sample(rng, ProbProg.Normal(s, σ, shape))
    return t
end

@testset "test" begin
    @testset "normal_hlo" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        code = @code_hlo optimize = false ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
        @test contains(repr(code), "enzyme.sample")
    end

    @testset "two_samples_hlo" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        code = @code_hlo optimize = false ProbProg.sample(rng, two_samples, μ, σ, shape)
        @test contains(repr(code), "enzyme.sample")
    end

    @testset "compose" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = false ProbProg.untraced_call(
            rng, compose, μ, σ, shape
        )
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog ProbProg.untraced_call(
            rng, compose, μ, σ, shape
        )
        @test !contains(repr(after), "enzyme.sample")
    end

    @testset "rng_state" begin
        shape = (10,)

        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        rng1 = ReactantRNG(copy(seed))

        _, X = @jit optimize = :probprog ProbProg.untraced_call(
            rng1, one_sample, μ, σ, shape
        )
        @test Array(rng1.seed) != Array(seed)

        rng2 = ReactantRNG(copy(seed))
        _, Y = @jit optimize = :probprog ProbProg.untraced_call(
            rng2, two_samples, μ, σ, shape
        )

        @test Array(rng2.seed) != Array(seed)
        @test Array(rng2.seed) != Array(rng1.seed)

        @test !all(Array(X) .≈ Array(Y))
    end
end
