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

function two_normals(rng, μ, σ, shape)
    x = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:x, logpdf=normal_logpdf)
    y = ProbProg.sample(rng, normal, x, σ, shape; symbol=:y, logpdf=normal_logpdf)
    return y
end

function nested_model(rng, μ, σ, shape)
    s = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    t = ProbProg.sample(rng, two_normals, s, σ, shape; symbol=:t)
    u = ProbProg.sample(rng, two_normals, t, σ, shape; symbol=:u)
    return u
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

        constraint = ProbProg.Constraint(:s => (fill(0.1, shape),))

        trace, weight = ProbProg.generate(rng, model, μ, σ, shape; constraint)

        @test trace.choices[:s][1] == constraint[ProbProg.Address(:s)][1]

        expected_weight =
            normal_logpdf(constraint[ProbProg.Address(:s)][1], 0.0, 1.0, shape) +
            normal_logpdf(
                trace.choices[:t][1], constraint[ProbProg.Address(:s)][1], 1.0, shape
            )
        @test weight ≈ expected_weight atol = 1e-6
    end

    @testset "composite addresses" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint = ProbProg.Constraint(
            :s => (fill(0.1, shape),),
            :t => :x => (fill(0.2, shape),),
            :u => :y => (fill(0.3, shape),),
        )

        trace, weight = ProbProg.generate(rng, nested_model, μ, σ, shape; constraint)

        @test trace.choices[:s][1] == fill(0.1, shape)
        @test trace.subtraces[:t].choices[:x][1] == fill(0.2, shape)
        @test trace.subtraces[:u].choices[:y][1] == fill(0.3, shape)

        s_weight = normal_logpdf(fill(0.1, shape), 0.0, 1.0, shape)
        tx_weight = normal_logpdf(fill(0.2, shape), fill(0.1, shape), 1.0, shape)
        ty_weight = normal_logpdf(
            trace.subtraces[:t].choices[:y][1], fill(0.2, shape), 1.0, shape
        )
        ux_weight = normal_logpdf(
            trace.subtraces[:u].choices[:x][1],
            trace.subtraces[:t].choices[:y][1],
            1.0,
            shape,
        )
        uy_weight = normal_logpdf(
            fill(0.3, shape), trace.subtraces[:u].choices[:x][1], 1.0, shape
        )

        expected_weight = s_weight + tx_weight + ty_weight + ux_weight + uy_weight
        @test weight ≈ expected_weight atol = 1e-6
    end

    @testset "compiled" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint1 = ProbProg.Constraint(:s => (fill(0.1, shape),))

        constrained_addresses = ProbProg.extract_addresses(constraint1)

        constraint_ptr1 = Reactant.ConcreteRNumber(
            reinterpret(UInt64, pointer_from_objref(constraint1))
        )

        wrapper_fn(rng, constraint_ptr, μ, σ) = ProbProg.generate_internal(
            rng, model, μ, σ, shape; constraint_ptr, constrained_addresses
        )

        compiled_fn = @compile optimize = :probprog wrapper_fn(rng, constraint_ptr1, μ, σ)

        trace1 = nothing
        seed_buffer = only(rng.seed.data).buffer
        GC.@preserve seed_buffer constraint1 begin
            trace1, _ = compiled_fn(rng, constraint_ptr1, μ, σ)

            while !isready(trace1)
                yield()
            end
        end
        trace1 = unsafe_pointer_to_objref(Ptr{Any}(Array(trace1)[1]))

        constraint2 = ProbProg.Constraint(:s => (fill(0.2, shape),))
        constraint_ptr2 = Reactant.ConcreteRNumber(
            reinterpret(UInt64, pointer_from_objref(constraint2))
        )

        trace2 = nothing
        seed_buffer = only(rng.seed.data).buffer
        GC.@preserve seed_buffer constraint2 begin
            trace2, _ = compiled_fn(rng, constraint_ptr2, μ, σ)

            while !isready(trace2)
                yield()
            end
        end
        trace2 = unsafe_pointer_to_objref(Ptr{Any}(Array(trace2)[1]))

        @test trace1.choices[:s][1] != trace2.choices[:s][1]
    end
end
