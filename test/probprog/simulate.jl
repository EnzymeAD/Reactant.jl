using Reactant, Test, Random
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function model(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(normal, rng, μ, σ, shape; symbol=:s)
    t = ProbProg.sample(normal, rng, s, σ, shape; symbol=:t)
    return t
end

@testset "Simulate" begin
    @testset "simulate_hlo" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = false ProbProg.simulate_internal(
            model, seed, μ, σ, shape; trace=ProbProg.ProbProgTrace()
        )
        @test contains(repr(before), "enzyme.simulate")

        after = @code_hlo optimize = :probprog ProbProg.simulate_internal(
            model, seed, μ, σ, shape; trace=ProbProg.ProbProgTrace()
        )
        @test !contains(repr(after), "enzyme.simulate")
    end

    @testset "normal_simulate" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace = ProbProg.simulate(model, seed, μ, σ, shape)

        @test size(trace.retval) == shape
        @test haskey(trace.choices, :s)
        @test haskey(trace.choices, :t)
        @test size(trace.choices[:s]) == shape
        @test size(trace.choices[:t]) == shape
    end

    @testset "correctness" begin
        op(x, y) = x * y'
        function fake_model(x, y)
            return ProbProg.sample(op, x, y; symbol=:matmul)
        end

        x = reshape(collect(Float64, 1:12), (4, 3))
        y = reshape(collect(Float64, 1:12), (4, 3))
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        trace = ProbProg.simulate(fake_model, x_ra, y_ra)

        @test Array(trace.retval) == op(x, y)
        @test haskey(trace.choices, :matmul)
        @test trace.choices[:matmul] == op(x, y)
    end
end
