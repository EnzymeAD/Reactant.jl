using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)

function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end

function model(rng, xs)
    _, param_a = ProbProg.sample(rng, ProbProg.Normal(0.0, 5.0, (1,)); symbol=:param_a)
    _, param_b = ProbProg.sample(rng, ProbProg.Normal(0.0, 5.0, (1,)); symbol=:param_b)

    _, ys_a = ProbProg.sample(
        rng, ProbProg.Normal(param_a .+ xs[1:5], 0.5, (5,)); symbol=:ys_a
    )

    _, ys_b = ProbProg.sample(
        rng, ProbProg.Normal(param_b .+ xs[6:10], 0.5, (5,)); symbol=:ys_b
    )

    return vcat(ys_a, ys_b)
end

function hmc_program(
    rng,
    model,
    xs,
    step_size,
    trajectory_length,
    num_warmup,
    num_samples,
    inverse_mass_matrix,
    constraint,
    constrained_addresses,
    selection,
    adapt_step_size,
    adapt_mass_matrix,
)
    t, _, _ = ProbProg.generate(rng, constraint, model, xs; constrained_addresses)

    t, accepted, _ = ProbProg.mcmc(
        rng,
        t,
        model,
        xs;
        selection,
        algorithm=:HMC,
        inverse_mass_matrix,
        step_size,
        trajectory_length,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )

    return t, accepted
end

function run_hmc_test(;
    adapt_step_size::Bool, adapt_mass_matrix::Bool, num_warmup::Int=10, num_samples::Int=5
)
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    xs = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ys_a = [-2.3, -1.6, -0.4, 0.6, 1.4]
    ys_b = [-2.6, -1.4, -0.6, 0.4, 1.6]

    obs = ProbProg.Constraint(
        :param_a => ([0.0],), :param_b => ([0.0],), :ys_a => (ys_a,), :ys_b => (ys_b,)
    )
    constrained_addresses = ProbProg.extract_addresses(obs)

    step_size = ConcreteRNumber(0.1)
    trajectory_length = 1.0
    inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])

    selection = ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b))

    compile_time_s = @elapsed begin
        compiled_fn = @compile optimize = :probprog hmc_program(
            rng,
            model,
            xs,
            step_size,
            trajectory_length,
            num_warmup,
            num_samples,
            inverse_mass_matrix,
            obs,
            constrained_addresses,
            selection,
            adapt_step_size,
            adapt_mass_matrix,
        )
    end
    println("Compile time: $(round(compile_time_s * 1000, digits=2)) ms")

    seed_buffer = only(rng.seed.data).buffer
    trace = nothing
    accepted = nothing

    GC.@preserve seed_buffer obs begin
        run_time_s = @elapsed begin
            trace, accepted = compiled_fn(
                rng,
                model,
                xs,
                step_size,
                trajectory_length,
                num_warmup,
                num_samples,
                inverse_mass_matrix,
                obs,
                constrained_addresses,
                selection,
                adapt_step_size,
                adapt_mass_matrix,
            )
            trace = ProbProg.ProbProgTrace(trace)
        end
        println("Run time: $(round(run_time_s * 1000, digits=2)) ms")
    end

    println("\nRESULTS")
    println("-"^70)
    println("accepted: $(Array(accepted))")
    println("param_a (all samples): $(only(trace.choices[:param_a]))")
    println("param_b (all samples): $(only(trace.choices[:param_b]))")
    println()

    return trace, accepted
end

@testset "hmc_adaptation_combinations" begin
    @testset "adapt_step_size=$ass, adapt_mass_matrix=$amm" for ass in [false, true],
        amm in [false, true]

        println("\n" * "="^70)
        println("TESTING: adapt_step_size=$ass, adapt_mass_matrix=$amm")
        println("="^70 * "\n")

        trace, accepted = run_hmc_test(;
            adapt_step_size=ass, adapt_mass_matrix=amm, num_warmup=20, num_samples=500
        )

        @test trace !== nothing
        @test accepted !== nothing
        @test haskey(trace.choices, :param_a)
        @test haskey(trace.choices, :param_b)
    end
end
