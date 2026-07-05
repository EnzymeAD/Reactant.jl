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
        guide=:AutoDelta,
        optimizer=:Adam,
        step_size,
        num_steps,
        num_particles=1,
        initial_position=init_loc,
    )
    return trace, elbo_traj, state.loc, state.scale
end

@testset "vi_autodelta_pointwise" begin
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
    lr = 0.01
    num_steps = 5

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

    impulse_loc_traj = Array(impulse_trace)
    impulse_elbo_traj = Array(impulse_elbo)

    @test size(impulse_loc_traj) == (num_steps, 1)
    @test size(impulse_elbo_traj) == (num_steps,)

    if check_numpyro_available()
        jnp = pyimport("jax.numpy")
        random_mod = pyimport("jax.random")
        numpyro_mod = pyimport("numpyro")
        dist = pyimport("numpyro.distributions")

        obs_y_py = obs_y
        numpyro_model = pyfunc(
            function ()
                x = numpyro_mod.sample("x", dist.Normal(0.0, 1.0))
                numpyro_mod.sample("y", dist.Normal(x, 0.5); obs=obs_y_py)
                return x
            end; name="model"
        )

        rng_key = seed_to_rbg_key(UInt64[0, 0])

        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            model_args=(),
            init_loc_values=Dict(:x => 0.0),
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            guide_kind=:AutoDelta,
            param_names=[:x_auto_loc],
        )

        compare_trajectories_pointwise(
            impulse_loc_traj,
            py_traj.loc_traj[:x_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="x_auto_loc (AutoDelta)",
        )

        @test all(impulse_scale .== 0.0)

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (AutoDelta)",
        )
    end
end
