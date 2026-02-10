using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

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

function nuts_program(
    rng,
    model,
    xs,
    step_size,
    max_tree_depth::Int,
    num_warmup::Int,
    num_samples::Int,
    inverse_mass_matrix,
    constraint_tensor,
    constrained_addresses,
    selection,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
)
    trace, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, xs; constrained_addresses
    )

    trace, diagnostics, _ = ProbProg.mcmc(
        rng,
        trace,
        model,
        xs;
        selection,
        algorithm=:NUTS,
        inverse_mass_matrix,
        step_size,
        max_tree_depth,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )

    return trace, diagnostics
end

function run_nuts_test(;
    adapt_step_size::Bool, adapt_mass_matrix::Bool, num_warmup::Int=10, num_samples::Int=5
)
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    xs = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ys_a = [-2.3, -1.6, -0.4, 0.6, 1.4]
    ys_b = [-2.6, -1.4, -0.6, 0.4, 1.6]

    obs = ProbProg.Constraint(
        :param_a => [0.0], :param_b => [0.0], :ys_a => ys_a, :ys_b => ys_b
    )
    constrained_addresses = ProbProg.extract_addresses(obs)

    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    step_size = ConcreteRNumber(0.1)
    max_tree_depth = 10
    inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])
    selection = ProbProg.select(ProbProg.Address(:param_a), ProbProg.Address(:param_b))

    compile_time_s = @elapsed begin
        compiled_fn, tt = ProbProg.with_trace() do
            @compile optimize = :probprog nuts_program(
                rng,
                model,
                xs,
                step_size,
                max_tree_depth,
                num_warmup,
                num_samples,
                inverse_mass_matrix,
                obs_tensor,
                constrained_addresses,
                selection,
                adapt_step_size,
                adapt_mass_matrix,
            )
        end
    end

    run_time_s = @elapsed begin
        trace_tensor, diagnostics = compiled_fn(
            rng,
            model,
            xs,
            step_size,
            max_tree_depth,
            num_warmup,
            num_samples,
            inverse_mass_matrix,
            obs_tensor,
            constrained_addresses,
            selection,
            adapt_step_size,
            adapt_mass_matrix,
        )
        trace_tensor = Array(trace_tensor)
        diagnostics = Array(diagnostics)
    end

    selected_entries = ProbProg.filter_entries_by_selection(tt.entries, selection)
    trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)

    return trace, diagnostics
end

@testset "nuts_adaptation_combinations" begin
    @testset "adapt_step_size=$ass, adapt_mass_matrix=$amm" for ass in [false, true],
        amm in [false, true]

        trace, diagnostics = run_nuts_test(;
            adapt_step_size=ass, adapt_mass_matrix=amm, num_warmup=200, num_samples=5
        )

        @test trace !== nothing
        @test diagnostics !== nothing
        @test haskey(trace.choices, :param_a)
        @test haskey(trace.choices, :param_b)

        @test size(trace.choices[:param_a]) == (5, 1)
        @test size(trace.choices[:param_b]) == (5, 1)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            numpyro_mod = pyimport("numpyro")
            dist = pyimport("numpyro.distributions")

            numpyro_model = pyfunc(
                function (xs, ys_a_obs, ys_b_obs)
                    param_a = numpyro_mod.sample("param_a", dist.Normal(0.0, 5.0))
                    param_b = numpyro_mod.sample("param_b", dist.Normal(0.0, 5.0))
                    pywith(numpyro_mod.plate("obs_a", 5)) do _
                        numpyro_mod.sample(
                            "ys_a", dist.Normal(param_a + xs[pyslice(5)], 0.5); obs=ys_a_obs
                        )
                    end
                    pywith(numpyro_mod.plate("obs_b", 5)) do _
                        numpyro_mod.sample(
                            "ys_b",
                            dist.Normal(param_b + xs[pyslice(5, pybuiltins.None)], 0.5);
                            obs=ys_b_obs,
                        )
                    end
                end;
                name="model",
            )

            xs_jnp = jnp.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
            ys_a_jnp = jnp.array([-2.3, -1.6, -0.4, 0.6, 1.4])
            ys_b_jnp = jnp.array([-2.6, -1.4, -0.6, 0.4, 1.6])

            numpyro_samples = run_numpyro_mcmc(;
                model=numpyro_model,
                rng_key=seed_to_rbg_key(UInt64[1, 5]),
                model_args=(xs_jnp, ys_a_jnp, ys_b_jnp),
                init_params=pydict(; param_a=jnp.array(0.0), param_b=jnp.array(0.0)),
                param_names=[:param_a, :param_b],
                algorithm=:NUTS,
                num_warmup=200,
                num_samples=5,
                dense_mass=true,
                inverse_mass_matrix=jnp.array(
                    pylist([pylist([0.5, 0.0]), pylist([0.0, 0.5])])
                ),
                adapt_step_size=ass,
                adapt_mass_matrix=amm,
            )

            compare_samples_pointwise(trace, numpyro_samples, [:param_a, :param_b])
        end
    end
end
