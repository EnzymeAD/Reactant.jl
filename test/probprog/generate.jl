using Reactant, Test, Random, Statistics
using Reactant: ProbProg, ReactantRNG

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    _, t = ProbProg.sample(rng, normal, s, σ, shape; symbol=:t, logpdf=normal_logpdf)
    return t
end

function two_normals(rng, μ, σ, shape)
    _, x = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:x, logpdf=normal_logpdf)
    _, y = ProbProg.sample(rng, normal, x, σ, shape; symbol=:y, logpdf=normal_logpdf)
    return y
end

function nested_model(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, normal, μ, σ, shape; symbol=:s, logpdf=normal_logpdf)
    _, t = ProbProg.sample(rng, two_normals, s, σ, shape; symbol=:t)
    _, u = ProbProg.sample(rng, two_normals, t, σ, shape; symbol=:u)
    return u
end

@testset "Generate" begin
    @testset "unconstrained" begin
        shape = (1000,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)
        trace, weight = ProbProg.generate_(rng, ProbProg.Constraint(), model, μ, σ, shape)
        @test mean(trace.retval[1]) ≈ 0.0 atol = 0.05 rtol = 0.05
    end

    @testset "constrained" begin
        shape = (10,)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        μ = Reactant.ConcreteRNumber(0.0)
        σ = Reactant.ConcreteRNumber(1.0)

        constraint = ProbProg.Constraint(:s => fill(0.1, shape))

        trace, weight = ProbProg.generate_(rng, constraint, model, μ, σ, shape)

        @test trace.choices[:s][1, :] == constraint[ProbProg.Address(:s)]

        expected_weight =
            normal_logpdf(constraint[ProbProg.Address(:s)], 0.0, 1.0, shape) +
            normal_logpdf(
                trace.choices[:t][1, :], constraint[ProbProg.Address(:s)], 1.0, shape
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
            :s => fill(0.1, shape),
            :t => :x => fill(0.2, shape),
            :u => :y => fill(0.3, shape),
        )

        trace, weight = ProbProg.generate_(rng, constraint, nested_model, μ, σ, shape)

        @test trace.choices[:s][1, :] == fill(0.1, shape)
        @test trace.subtraces[:t].choices[:x][1, :] == fill(0.2, shape)
        @test trace.subtraces[:u].choices[:y][1, :] == fill(0.3, shape)

        s_weight = normal_logpdf(fill(0.1, shape), 0.0, 1.0, shape)
        tx_weight = normal_logpdf(fill(0.2, shape), fill(0.1, shape), 1.0, shape)
        ty_weight = normal_logpdf(
            trace.subtraces[:t].choices[:y][1, :], fill(0.2, shape), 1.0, shape
        )
        ux_weight = normal_logpdf(
            trace.subtraces[:u].choices[:x][1, :],
            trace.subtraces[:t].choices[:y][1, :],
            1.0,
            shape,
        )
        uy_weight = normal_logpdf(
            fill(0.3, shape), trace.subtraces[:u].choices[:x][1, :], 1.0, shape
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

        constraint1 = ProbProg.Constraint(:s => fill(0.1, shape))
        constrained_addresses = ProbProg.extract_addresses(constraint1)

        c1_flat = Float64[]
        for addr in constrained_addresses
            append!(c1_flat, vec(constraint1[addr]))
        end
        c1_tensor = Reactant.to_rarray(reshape(c1_flat, 1, :))

        tt = ProbProg.TracedTrace()
        compiled_fn = ScopedValues.with(ProbProg.TRACING_TRACE => tt) do
            @compile optimize = :probprog ProbProg.generate(
                rng, c1_tensor, model, μ, σ, shape; constrained_addresses
            )
        end
        t1, w1, r1 = compiled_fn(rng, c1_tensor, model, μ, σ, shape)
        trace1 = ProbProg.unflatten_trace(t1, w1, tt.entries, r1[2:end])

        c2_flat = fill(0.2, 1, length(c1_flat))
        c2_tensor = Reactant.to_rarray(reshape(c2_flat, 1, :))

        t2, w2, r2 = compiled_fn(rng, c2_tensor, model, μ, σ, shape)
        trace2 = ProbProg.unflatten_trace(t2, w2, tt.entries, r2[2:end])

        @test trace1.choices[:s][1, :] != trace2.choices[:s][1, :]
    end
end
