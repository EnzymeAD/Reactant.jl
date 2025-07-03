using Reactant, Test, Random
using Reactant: ProbProg

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
normal_logpdf(x, μ, σ, _) = -sum(log.(σ)) - sum((μ .- x) .^ 2) / (2 * σ^2)

function product_two_normals(rng, μ, σ, shape)
    a = ProbProg.sample(normal, rng, μ, σ, shape; symbol=:a, logpdf=normal_logpdf)
    b = ProbProg.sample(normal, rng, μ, σ, shape; symbol=:b, logpdf=normal_logpdf)
    return a .* b
end

function model(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(normal, rng, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    t = ProbProg.sample(normal, rng, s, σ, shape; symbol=:t, logpdf=normal_logpdf)
    return t
end

function model2(seed, μ, σ, shape)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    s = ProbProg.sample(product_two_normals, rng, μ, σ, shape; symbol=:s)
    t = ProbProg.sample(product_two_normals, rng, s, σ, shape; symbol=:t)
    return t
end

@testset "Simulate" begin
    @testset "hlo" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = false ProbProg.simulate_internal(
            model, seed, μ, σ, shape
        )
        @test contains(repr(before), "enzyme.simulate")

        after = @code_hlo optimize = :probprog ProbProg.simulate_internal(
            model, seed, μ, σ, shape
        )
        @test !contains(repr(after), "enzyme.simulate")
    end

    @testset "normal_simulate" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace, weight = ProbProg.simulate(model, seed, μ, σ, shape)

        @test size(trace.retval) == shape
        @test haskey(trace.choices, :s)
        @test haskey(trace.choices, :t)
        @test size(trace.choices[:s]) == shape
        @test size(trace.choices[:t]) == shape
        @test trace.weight isa Float64
    end

    @testset "simple_fake" begin
        op(x, y) = x * y'
        logpdf(res, _, _) = sum(res)
        function fake_model(x, y)
            return ProbProg.sample(op, x, y; symbol=:matmul, logpdf=logpdf)
        end

        x = reshape(collect(Float64, 1:12), (4, 3))
        y = reshape(collect(Float64, 1:12), (4, 3))
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        trace, weight = ProbProg.simulate(fake_model, x_ra, y_ra)

        @test Array(trace.retval) == op(x, y)
        @test haskey(trace.choices, :matmul)
        @test trace.choices[:matmul] == op(x, y)
        @test trace.weight == logpdf(op(x, y), x, y)
    end

    @testset "submodel_fake" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace, weight = ProbProg.simulate(model2, seed, μ, σ, shape)
        
        println(trace)

        @test size(trace.retval) == shape
        @test haskey(trace.choices, :s)
        @test haskey(trace.choices, :t)
        @test size(trace.choices[:s]) == shape
        @test size(trace.choices[:t]) == shape
        @test trace.weight isa Float64
    end
end
