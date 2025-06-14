using Reactant, Test, Random, StableRNGs, Statistics
using Reactant: ProbProg

@testset "Simulate" begin
    normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

    function simulate_model(trace, seed, μ, σ, shape)
        function model(seed, μ, σ, shape)
            rng = Random.default_rng()
            Random.seed!(rng, seed)
            s = ProbProg.sample!(normal, rng, μ, σ, shape; symbol=:s)
            t = ProbProg.sample!(normal, rng, s, σ, shape; symbol=:t)
            return t
        end

        result = ProbProg.simulate!(model, seed, μ, σ, shape; trace)
        return result
    end
    @testset "normal_hlo" begin
        shape = (10000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace = ProbProg.createTrace()

        before = @code_hlo optimize = :no_enzyme simulate_model(trace, seed, μ, σ, shape)
        @test contains(repr(before), "enzyme.simulate")
        @test contains(repr(before), "enzyme.sample")

        after = @code_hlo optimize = :probprog simulate_model(trace, seed, μ, σ, shape)
        @test !contains(repr(after), "enzyme.simulate")
        @test !contains(repr(after), "enzyme.sample")
        @test contains(repr(after), "enzyme_probprog_add_sample_to_trace")
    end

    @testset "normal_simulate" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace = ProbProg.createTrace()

        result = Array(
            @jit optimize = :probprog sync = true simulate_model(trace, seed, μ, σ, shape)
        )

        ProbProg.print_trace(trace)
        @test size(result) == shape
    end
end
