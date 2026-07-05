using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

function model(rng)
    _, x = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:x)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(x, 0.5); symbol=:y)
    return x
end

function svi_program(
    rng,
    model,
    step_size,
    num_steps::Int,
    selection,
    obs_tensor,
    constrained_addresses,
    init_loc,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng,
        trace0,
        model;
        selection,
        guide=:AutoNormal,
        optimizer=:Adam,
        step_size,
        num_steps,
        num_particles=1,
        init_scale=0.1,
        initial_position=init_loc,
    )
    return trace, elbo_traj, state.loc, state.scale
end

function run_svi_test(; num_steps::Int=5, lr::Float64=0.01)
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs_y = 1.0
    # Constraining every site keeps generate from consuming rng (NumPyro parity).
    obs = ProbProg.Constraint(:x => [0.0], :y => [obs_y])
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:x))
    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, 1))

    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_program(
            rng,
            model,
            step_size,
            num_steps,
            selection,
            obs_tensor,
            constrained_addresses,
            init_loc,
        )
    end

    impulse_trace, impulse_elbo, impulse_loc, impulse_scale = compiled(
        rng,
        model,
        step_size,
        num_steps,
        selection,
        obs_tensor,
        constrained_addresses,
        init_loc,
    )

    return (
        loc_trajectory=Array(impulse_trace),
        elbo_trajectory=Array(impulse_elbo),
        final_loc=Array(impulse_loc),
        final_scale=Array(impulse_scale),
        seed=seed,
        obs_y=obs_y,
        num_steps=num_steps,
        lr=lr,
    )
end

@testset "vi_pointwise_comparison" begin
    result = run_svi_test(; num_steps=5, lr=0.01)

    @test size(result.loc_trajectory) == (5, 1)
    @test size(result.elbo_trajectory) == (5,)

    if check_numpyro_available()
        jnp = pyimport("jax.numpy")
        random_mod = pyimport("jax.random")
        numpyro_mod = pyimport("numpyro")
        dist = pyimport("numpyro.distributions")

        obs_y_py = result.obs_y
        numpyro_model = pyfunc(
            function ()
                x = numpyro_mod.sample("x", dist.Normal(0.0, 1.0))
                numpyro_mod.sample("y", dist.Normal(x, 0.5); obs=obs_y_py)
                return x
            end; name="model"
        )

        # RBG key so NumPyro uses XLA's RngBitGenerator, as stablehlo.rng_bit_generator.
        rng_key = seed_to_rbg_key(UInt64[0, 0])

        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            model_args=(),
            init_loc_values=Dict(:x => 0.0),
            init_scale=0.1,
            num_steps=result.num_steps,
            step_size=result.lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:x_auto_loc, :x_auto_scale],
        )

        impulse_loc_flat = vec(result.loc_trajectory)
        compare_trajectories_pointwise(
            result.loc_trajectory,
            py_traj.loc_traj[:x_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="x_auto_loc",
        )

        # Only the final scale is compared: Impulse exports no per-step scale trajectory.
        impulse_final_scale = result.final_scale[1]
        numpyro_final_scale = py_traj.loc_traj[:x_auto_scale][end]
        @info "final_scale: impulse=$impulse_final_scale numpyro=$numpyro_final_scale"
        @test isapprox(impulse_final_scale, numpyro_final_scale; atol=1e-12, rtol=1e-12)

        compare_trajectories_pointwise(
            reshape(result.elbo_trajectory, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj",
        )
    end
end
