using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

# Reference: https://www.gen.dev/docs/stable/getting_started/linear_regression/

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -sum(log.(σ)) - length(x) / 2 * log(2π) - sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function my_model(rng, xs)
    slope = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    intercept = ProbProg.sample(
        rng, normal, 0.0, 10.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )

    ys = ProbProg.sample(
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

function my_inference_program(xs, ys, num_iters)
    xs_r = Reactant.to_rarray(xs)

    observations = ProbProg.Constraint(:ys => (ys,))

    seed = Reactant.to_rarray(UInt64[1, 4])
    rng = ReactantRNG(seed)

    trace, _ = ProbProg.generate(rng, my_model, xs_r; constraint=observations)

    trace = ProbProg.with_compiled_cache() do cache
        local t = trace
        for _ in 1:num_iters
            t, _ = ProbProg.metropolis_hastings(
                t, ProbProg.select(:slope); compiled_cache=cache
            )
            t, _ = ProbProg.metropolis_hastings(
                t, ProbProg.select(:intercept); compiled_cache=cache
            )
        end
        return t
    end

    choices = ProbProg.get_choices(trace)
    return (Array(choices[:slope][1])[1], Array(choices[:intercept][1])[1])
end

@testset "linear_regression" begin
    @testset "simulate" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        xs_r = Reactant.to_rarray(xs)

        trace, _ = ProbProg.simulate(rng, my_model, xs_r)

        @test haskey(trace.choices, :slope)
        @test haskey(trace.choices, :intercept)
        @test haskey(trace.choices, :ys)
    end

    @testset "inference" begin
        Random.seed!(1) # For Julia side RNG
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

        slope, intercept = my_inference_program(xs, ys, 10000)

        @show slope, intercept

        @test slope ≈ -2.0 rtol = 0.05
        @test intercept ≈ 10.0 rtol = 0.05
    end
end
