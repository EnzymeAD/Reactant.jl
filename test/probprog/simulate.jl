using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -sum(log.(σ)) - length(x) / 2 * log(2π) - sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function product_two_normals(rng, μ, σ, shape)
    a = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:a, logpdf=normal_logpdf)
    b = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:b, logpdf=normal_logpdf)
    return a .* b
end

function model(rng, μ, σ, shape)
    s = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    t = ProbProg.sample(rng, normal, s, σ, shape; symbol=:t, logpdf=normal_logpdf)
    return t
end

function model2(rng, μ, σ, shape)
    s = ProbProg.sample(rng, product_two_normals, μ, σ, shape; symbol=:s)
    t = ProbProg.sample(rng, product_two_normals, s, σ, shape; symbol=:t)
    return t
end

@testset "Simulate" begin
    @testset "hlo" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        before = @code_hlo optimize = false ProbProg.simulate_internal(
            rng, model, μ, σ, shape
        )
        @test contains(repr(before), "enzyme.simulate")

        unlowered = @code_hlo optimize = :probprog_no_lowering ProbProg.simulate_internal(
            rng, model, μ, σ, shape
        )
        @test !contains(repr(unlowered), "enzyme.simulate")
        @test contains(repr(unlowered), "enzyme.addSampleToTrace")
        @test contains(repr(unlowered), "enzyme.addWeightToTrace")
        @test contains(repr(unlowered), "enzyme.addRetvalToTrace")

        after = @code_hlo optimize = :probprog ProbProg.simulate_internal(
            rng, model, μ, σ, shape
        )
        @test !contains(repr(after), "enzyme.simulate")
        @test !contains(repr(after), "enzyme.addSampleToTrace")
        @test !contains(repr(after), "enzyme.addWeightToTrace")
        @test !contains(repr(after), "enzyme.addRetvalToTrace")
    end

    @testset "normal_simulate" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace, weight = ProbProg.simulate(rng, model, μ, σ, shape)
        println(trace)

        @test size(trace.retval[1]) == shape
        @test haskey(trace.choices, :s)
        @test haskey(trace.choices, :t)
        @test size(trace.choices[:s][1]) == shape
        @test size(trace.choices[:t][1]) == shape
        @test trace.weight isa Float64
    end

    @testset "simple_fake" begin
        op(_, x, y) = x * y'
        logpdf(res, _, _) = sum(res)
        function fake_model(rng, x, y)
            return ProbProg.sample(rng, op, x, y; symbol=:matmul, logpdf=logpdf)
        end

        x = reshape(collect(Float64, 1:12), (4, 3))
        y = reshape(collect(Float64, 1:12), (4, 3))
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)

        trace, weight = ProbProg.simulate(rng, fake_model, x_ra, y_ra)

        @test Array(trace.retval[1]) == op(rng, x, y)
        @test haskey(trace.choices, :matmul)
        @test trace.choices[:matmul][1] == op(rng, x, y)
        @test trace.weight == logpdf(op(rng, x, y), x, y)
    end

    @testset "submodel_fake" begin
        shape = (3, 3, 3)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        trace, weight = ProbProg.simulate(rng, model2, μ, σ, shape)

        @test size(trace.retval[1]) == shape

        @test length(trace.choices) == 2
        @test haskey(trace.choices, :s)
        @test haskey(trace.choices, :t)

        @test length(trace.subtraces) == 2
        @test haskey(trace.subtraces[:s].choices, :a)
        @test haskey(trace.subtraces[:s].choices, :b)
        @test haskey(trace.subtraces[:t].choices, :a)
        @test haskey(trace.subtraces[:t].choices, :b)

        @test size(trace.choices[:s][1]) == shape
        @test size(trace.choices[:t][1]) == shape

        @test trace.weight isa Float64

        @test trace.weight ≈ trace.subtraces[:s].weight + trace.subtraces[:t].weight
    end
end
