using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, xs)
    _, param_a = ProbProg.sample(
        rng, normal, 0.0, 5.0, (1,); symbol=:param_a, logpdf=normal_logpdf
    )
    _, param_b = ProbProg.sample(
        rng, normal, 0.0, 5.0, (1,); symbol=:param_b, logpdf=normal_logpdf
    )

    _, ys_a = ProbProg.sample(
        rng, normal, param_a .+ xs[1:5], 0.5, (5,); symbol=:ys_a, logpdf=normal_logpdf
    )

    _, ys_b = ProbProg.sample(
        rng, normal, param_b .+ xs[6:10], 0.5, (5,); symbol=:ys_b, logpdf=normal_logpdf
    )

    return vcat(ys_a, ys_b)
end

function hmc_program(rng, t, model, xs, step_size, num_steps, mass, initial_momentum)
    trace_ptr_val = reinterpret(UInt64, pointer_from_objref(t))
    trace_ptr = Reactant.Ops.fill(trace_ptr_val, Int64[])

    trace_ptr, accepted, _ = ProbProg.hmc(
        rng,
        trace_ptr,
        model,
        xs;
        selection=ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b)),
        mass=mass,
        step_size=step_size,
        num_steps=num_steps,
        initial_momentum=initial_momentum,
    )

    return trace_ptr, accepted
end

@testset "hmc" begin
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    xs = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ys_a = [-2.3, -1.6, -0.4, 0.6, 1.4]
    ys_b = [-2.6, -1.4, -0.6, 0.4, 1.6]
    obs = ProbProg.Constraint(
        :param_a => ([0.0],), :param_b => ([0.0],), :ys_a => (ys_a,), :ys_b => (ys_b,)
    )
    init_trace, _ = ProbProg.generate_(rng, model, xs; constraint=obs)

    step_size = ConcreteRNumber(0.01)
    num_steps_compile = ConcreteRNumber(10)
    num_steps_run = ConcreteRNumber(20)
    mass = nothing
    initial_momentum = ConcreteRArray([0.0, 0.0])

    code = @code_hlo optimize = :probprog hmc_program(
        rng, init_trace, model, xs, step_size, num_steps_compile, mass, initial_momentum
    )
    @test contains(repr(code), "enzyme_probprog_get_flattened_samples_from_trace")
    @test contains(repr(code), "enzyme_probprog_get_weight_from_trace")
    @test !contains(repr(code), "enzyme.mh")
    @test !contains(repr(code), "enzyme.mcmc")

    compiled_fn = @compile optimize = :probprog hmc_program(
        rng, init_trace, model, xs, step_size, num_steps_compile, mass, initial_momentum
    )

    trace = init_trace
    seed_buffer = only(rng.seed.data).buffer
    GC.@preserve seed_buffer init_trace begin
        GC.enable(false)
        try
            trace_ptr, _ = compiled_fn(
                rng, trace, model, xs, step_size, num_steps_run, mass, initial_momentum
            )
            while !isready(trace_ptr)
                yield()
            end
            trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace_ptr)[1]))
        finally
            GC.enable(true)
            GC.gc()
        end
    end

    # NumPyro results
    @test only(trace.choices[:param_a])[1] ≈ 0.7630458236301266 rtol = 1e-6
    @test only(trace.choices[:param_b])[1] ≈ -1.1296070526289126 rtol = 1e-6
end
