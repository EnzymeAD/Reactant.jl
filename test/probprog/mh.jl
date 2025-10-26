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

function mh_program(rng, model, xs, num_iters, constraint_ptr, constrained_addresses)
    init_trace, _, _ = ProbProg.generate(
        rng,
        model,
        xs;
        constraint_ptr=constraint_ptr,
        constrained_addresses=constrained_addresses,
    )

    trace_ptr = init_trace
    @trace for _ in 1:num_iters
        trace_ptr, _ = ProbProg.mh(
            rng, trace_ptr, model, xs; selection=ProbProg.select(ProbProg.Address(:slope))
        )
        trace_ptr, _ = ProbProg.mh(
            rng,
            trace_ptr,
            model,
            xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)),
        )
    end

    return trace_ptr
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
        obs = ProbProg.Constraint(:ys => (ys,))
        num_iters = ConcreteRNumber(10000)
        constrained_addresses = ProbProg.extract_addresses(obs)
        constraint_ptr = ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(obs)))

        code = @code_hlo optimize = :probprog mh_program(
            rng, model, xs, 10000, constraint_ptr, constrained_addresses
        )
        @test contains(repr(code), "enzyme_probprog_get_sample_from_trace")
        @test contains(repr(code), "enzyme_probprog_get_weight_from_trace")
        @test !contains(repr(code), "enzyme.mh")

        compiled_fn = @compile optimize = :probprog mh_program(
            rng, model, xs, num_iters, constraint_ptr, constrained_addresses
        )

        trace = nothing
        seed_buffer = only(rng.seed.data).buffer
        num_iters = ConcreteRNumber(1000)
        GC.@preserve seed_buffer obs begin
            trace_ptr = compiled_fn(
                rng, model, xs, num_iters, constraint_ptr, constrained_addresses
            )
            trace = ProbProg.from_trace_tensor(trace_ptr)
        end

        slope = only(trace.choices[:slope])[1]
        intercept = only(trace.choices[:intercept])[1]
        @show slope, intercept

        @test slope ≈ -2.0 rtol = 0.1
        @test intercept ≈ 10.0 rtol = 0.1
    end
end
