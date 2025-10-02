using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG

# Reference: https://www.gen.dev/docs/stable/getting_started/linear_regression/

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -sum(log.(σ)) - length(x) / 2 * log(2π) - sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
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

function mh_program(rng, t, model, xs, num_iters)
    trace_ptr_val = reinterpret(UInt64, pointer_from_objref(t))
    trace_ptr = Reactant.Ops.fill(trace_ptr_val, Int64[])

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
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
        obs = ProbProg.Constraint(:ys => (ys,))
        init_trace, _ = ProbProg.generate_(rng, model, xs; constraint=obs)

        code = @code_hlo optimize = false mh_program(rng, init_trace, model, xs, 10000)
        println(code)

        seed_buffer = only(rng.seed.data).buffer
        GC.@preserve seed_buffer init_trace begin
            trace_ptr = @compile optimize = :probprog mh_program(
                rng, init_trace, model, xs, 10000
            )

            while !isready(trace_ptr)
                yield()
            end

            trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace_ptr)[1]))
        end

        slope = trace.choices[:slope][1]
        intercept = trace.choices[:intercept][1]
        @show slope, intercept

        @test slope ≈ -2.0 rtol = 0.05
        @test intercept ≈ 10.0 rtol = 0.05
    end
end