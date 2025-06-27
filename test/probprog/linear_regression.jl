using Reactant, Test, Random
using Reactant: ProbProg

# Reference: https://www.gen.dev/docs/stable/getting_started/linear_regression/

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
normal_logpdf(x, μ, σ, _) = -sum(log.(σ)) - sum((μ .- x) .^ 2) / (2 * σ^2)

function my_model(seed, xs)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    slope = ProbProg.sample(
        normal, rng, 0.0, 2.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    intercept = ProbProg.sample(
        normal, rng, 0.0, 10.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )

    ys = ProbProg.sample(
        normal,
        rng,
        slope .* xs .+ intercept,
        1.0,
        (length(xs),);
        symbol=:ys,
        logpdf=normal_logpdf,
    )

    return rng.seed, ys
end

function my_inference_program(xs, ys, num_iters)
    xs_r = Reactant.to_rarray(xs)

    constraints = ProbProg.choicemap()
    constraints[:ys] = [ys]

    seed = Reactant.to_rarray(UInt64[1, 4])

    trace, _ = ProbProg.generate(my_model, seed, xs_r; constraints)
    trace.args = (trace.retval[1], trace.args[2:end]...) # TODO: this is a temporary hack

    for i in 1:num_iters
        trace, _ = ProbProg.metropolis_hastings(trace, ProbProg.select(:slope))
        trace, _ = ProbProg.metropolis_hastings(trace, ProbProg.select(:intercept))
        choices = ProbProg.get_choices(trace)
        # @show i, choices[:slope], choices[:intercept]
    end

    choices = ProbProg.get_choices(trace)
    return (choices[:slope], choices[:intercept])
end

@testset "linear_regression" begin
    @testset "simulate" begin
        seed = Reactant.to_rarray(UInt64[1, 4])
        Random.seed!(42) # For Julia side RNG

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        xs_r = Reactant.to_rarray(xs)

        trace = ProbProg.simulate(my_model, seed, xs_r)

        @test haskey(trace.choices, :slope)
        @test haskey(trace.choices, :intercept)
        @test haskey(trace.choices, :ys)
    end

    @testset "inference" begin
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

        slope, intercept = my_inference_program(xs, ys, 5)

        # @show slope, intercept
    end
end
