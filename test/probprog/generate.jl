using Reactant, Test, Random, Statistics
using Reactant: ProbProg, ReactantRNG

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -sum(log.(σ)) - length(x) / 2 * log(2π) - sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, μ, σ, shape)
    s = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    t = ProbProg.sample(rng, normal, s, σ, shape; symbol=:t, logpdf=normal_logpdf)
    return t
end

@testset "Generate" begin
    @testset "unconstrained" begin
        shape = (1000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        trace, weight = ProbProg.generate(rng, model, μ, σ, shape)
        @test mean(trace.retval[1]) ≈ 0.0 atol = 0.05 rtol = 0.05
    end

    @testset "constrained" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint = Dict{Symbol,Any}(:s => (fill(0.1, shape),))

        trace, weight = ProbProg.generate(rng, model, μ, σ, shape; constraint)

        @test trace.choices[:s][1] == constraint[:s][1]

        expected_weight =
            normal_logpdf(constraint[:s][1], 0.0, 1.0, shape) +
            normal_logpdf(trace.choices[:t][1], constraint[:s][1], 1.0, shape)
        @test weight ≈ expected_weight atol = 1e-6
    end

    @testset "compiled" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint1 = Dict{Symbol,Any}(:s => (fill(0.1, shape),))

        constrained_symbols = Set(keys(constraint1))

        constraint_ptr1 = Reactant.ConcreteRNumber(
            reinterpret(UInt64, pointer_from_objref(constraint1))
        )

        wrapper_fn(constraint_ptr, rng, μ, σ) = ProbProg.generate_internal(
            rng, model, μ, σ, shape; constraint_ptr, constrained_symbols
        )

        compiled_fn = @compile optimize = :probprog wrapper_fn(constraint_ptr1, rng, μ, σ)

        trace1, weight = compiled_fn(constraint_ptr1, rng, μ, σ)
        trace1 = unsafe_pointer_to_objref(Ptr{Any}(Array(trace1)[1]))

        constraint2 = Dict{Symbol,Any}(:s => (fill(0.2, shape),))
        constraint_ptr2 = Reactant.ConcreteRNumber(
            reinterpret(UInt64, pointer_from_objref(constraint2))
        )

        trace2, _ = compiled_fn(constraint_ptr2, rng, μ, σ)
        trace2 = unsafe_pointer_to_objref(Ptr{Any}(Array(trace2)[1]))

        @test trace1.choices[:s][1] != trace2.choices[:s][1]
    end
end
