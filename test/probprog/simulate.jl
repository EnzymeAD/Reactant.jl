using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg
using Libdl: Libdl

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function simulate_model(seed, μ, σ, shape)
    function model(seed, μ, σ, shape)
        rng = Random.default_rng()
        Random.seed!(rng, seed)
        s = ProbProg.sample!(normal, rng, μ, σ, shape; symbol=:s)
        t = ProbProg.sample!(normal, rng, s, σ, shape; symbol=:t)
        return t
    end

    return ProbProg.simulate!(model, seed, μ, σ, shape)
end

@testset "Simulate" begin
    @testset "normal_hlo" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRArray(0.0)
        σ = Reactant.ConcreteRArray(1.0)

        before = @code_hlo optimize = :no_enzyme simulate_model(seed, μ, σ, shape)
        @test contains(repr(before), "enzyme.simulate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog simulate_model(seed, μ, σ, shape)
        @test !contains(repr(after), "enzyme.simulate")
        @test !contains(repr(after), "enzyme.sample")
        @test contains(repr(after), "enzyme_probprog_add_sample_to_trace")
        @test contains(repr(after), "enzyme_probprog_init_trace")
    end

    @testset "normal_simulate" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRArray(0.0)
        σ = Reactant.ConcreteRArray(1.0)
        X = ProbProg.getTrace(@jit optimize = :probprog simulate_model(seed, μ, σ, shape))
        @test X[:_integrity_check] == 0x123456789abcdef
        ProbProg.print_trace(X)
    end
end
