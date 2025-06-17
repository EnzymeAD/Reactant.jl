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

        trace = ProbProg.create_trace()

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

        trace = ProbProg.create_trace()

        result = Array(@jit optimize = :probprog simulate_model(trace, seed, μ, σ, shape))

        @test size(result) == shape
        @test haskey(trace, :s)
        @test haskey(trace, :t)
        @test size(trace[:s]) == shape
        @test size(trace[:t]) == shape
    end

    @testset "correctness" begin
        op(x, y) = x * y'
        function fake_model(x, y)
            return ProbProg.sample!(op, x, y; symbol=:matmul)
        end

        trace = ProbProg.create_trace()
        x = reshape(collect(Float64, 1:12), (4, 3))
        y = reshape(collect(Float64, 1:12), (4, 3))
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test Array(
            @jit optimize = :probprog ProbProg.simulate!(fake_model, x_ra, y_ra; trace)
        ) == op(x, y)

        @test haskey(trace, :matmul)
        @test trace[:matmul] == op(x, y)
    end
end
