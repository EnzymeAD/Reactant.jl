using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

# Reference: https://www.gen.dev/docs/stable/getting_started/linear_regression/

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, xs)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 10.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )

    _, ys = ProbProg.sample(
        rng,
        normal,
        slope .* xs .+ intercept,
        1.0,
        (length(xs),);
        symbol=:ys,
        logpdf=normal_logpdf,
    )

    return ys
end

function mh_program(rng, model, xs, num_iters, constraint_tensor, constrained_addresses)
    trace, weight, _ = ProbProg.generate(
        rng, constraint_tensor, model, xs; constrained_addresses=constrained_addresses
    )

    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:slope)),
        )
        trace, weight, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
        )
    end

    return trace, weight
end

@testset "linear_regression" begin
    @testset "simulate" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        xs_r = Reactant.to_rarray(xs)

        trace, _ = ProbProg.simulate_(rng, model, xs_r)

        @test haskey(trace.choices, :slope)
        @test haskey(trace.choices, :intercept)
        @test haskey(trace.choices, :ys)
    end

    @testset "inference" begin
        seed = Reactant.to_rarray(UInt64[1, 5])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
        obs = ProbProg.Constraint(:ys => ys)
        num_iters = ConcreteRNumber(10000)
        constrained_addresses = ProbProg.extract_addresses(obs)

        obs_flat = Float64[]
        for addr in constrained_addresses
            append!(obs_flat, vec(obs[addr]))
        end
        obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

        code, _ = ProbProg.with_trace() do
            @code_hlo optimize = :probprog mh_program(
                rng, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end
        @test !contains(repr(code), "enzyme.mh")

        num_iters = ConcreteRNumber(1000)
        compiled_fn, tt = ProbProg.with_trace() do
            @compile optimize = :probprog mh_program(
                rng, model, xs, num_iters, obs_tensor, constrained_addresses
            )
        end

        trace_tensor, weight_val = compiled_fn(
            rng, model, xs, num_iters, obs_tensor, constrained_addresses
        )
        trace = ProbProg.unflatten_trace(trace_tensor, weight_val, tt.entries, ())

        slope = trace.choices[:slope][1, 1]
        intercept = trace.choices[:intercept][1, 1]
        @test slope ≈ -2.0 rtol = 0.1
        @test intercept ≈ 10.0 rtol = 0.1
    end
end
