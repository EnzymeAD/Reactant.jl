using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, Profiler

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

function hmc_program(
    rng,
    model,
    xs,
    step_size,
    num_steps,
    inverse_mass_matrix,
    initial_momentum,
    constraint,
    constrained_addresses,
)
    t, _, _ = ProbProg.generate(rng, constraint, model, xs; constrained_addresses)

    t, accepted, _ = ProbProg.mcmc(
        rng,
        t,
        model,
        xs;
        selection=ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b)),
        algorithm=:HMC,
        inverse_mass_matrix,
        step_size,
        num_steps,
        initial_momentum,
    )

    return t, accepted
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
    constrained_addresses = ProbProg.extract_addresses(obs)

    step_size = ConcreteRNumber(0.001)
    num_steps_compile = ConcreteRNumber(1000)
    num_steps_run = ConcreteRNumber(40000000)
    inverse_mass_matrix = ConcreteRArray([1.0 0.0; 0.0 1.0])
    initial_momentum = ConcreteRArray([0.0, 0.0])

    code = @code_hlo optimize = :probprog hmc_program(
        rng,
        model,
        xs,
        step_size,
        num_steps_compile,
        inverse_mass_matrix,
        initial_momentum,
        obs,
        constrained_addresses,
    )
    @test contains(repr(code), "enzyme_probprog_get_flattened_samples_from_trace")
    @test contains(repr(code), "enzyme_probprog_get_weight_from_trace")
    @test !contains(repr(code), "enzyme.mcmc")

    compile_time_s = @elapsed begin
        compiled_fn = @compile optimize = :probprog hmc_program(
            rng,
            model,
            xs,
            step_size,
            num_steps_compile,
            inverse_mass_matrix,
            initial_momentum,
            obs,
            constrained_addresses,
        )
    end
    println("HMC compile time: $(round(compile_time_s * 1000, digits=2)) ms")

    seed_buffer = only(rng.seed.data).buffer
    trace = nothing
    enable_profiling = false

    GC.@preserve seed_buffer obs begin
        run_time_s = @elapsed begin
            if enable_profiling
                Profiler.with_profiler("./traces"; create_perfetto_link=true) do
                    trace, _ = compiled_fn(
                        rng,
                        model,
                        xs,
                        step_size,
                        num_steps_run,
                        inverse_mass_matrix,
                        initial_momentum,
                        obs,
                        constrained_addresses,
                    )
                end
            else
                trace, _ = compiled_fn(
                    rng,
                    model,
                    xs,
                    step_size,
                    num_steps_run,
                    inverse_mass_matrix,
                    initial_momentum,
                    obs,
                    constrained_addresses,
                )
            end
            trace = ProbProg.ProbProgTrace(trace)
        end
        println("HMC run time: $(round(run_time_s * 1000, digits=2)) ms")
    end

    # NumPyro results
    @test only(trace.choices[:param_a])[1] ≈ 0.01327671 rtol = 1e-6
    @test only(trace.choices[:param_b])[1] ≈ -0.01965474 rtol = 1e-6
end
