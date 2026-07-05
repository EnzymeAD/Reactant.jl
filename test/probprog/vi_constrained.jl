using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

function model(rng)
    _, sigma = ProbProg.sample(rng, ProbProg.LogNormal(0.0, 1.0); symbol=:sigma)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(0.0, sigma); symbol=:y)
    return sigma
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

@testset "vi_constrained_pointwise" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs_y = 1.0
    obs = ProbProg.Constraint(:sigma => [1.0], :y => [obs_y])
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:sigma))
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
                sigma = numpyro_mod.sample("sigma", dist.LogNormal(0.0, 1.0))
                numpyro_mod.sample("y", dist.Normal(0.0, sigma); obs=obs_y_py)
                return sigma
            end; name="model"
        )

        rng_key = seed_to_rbg_key(UInt64[0, 0])

        # Init unconstrained loc at 0 → sigma starts at exp(0) = 1.0.
        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            model_args=(),
            init_loc_values=Dict(:sigma => 1.0),
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:sigma_auto_loc, :sigma_auto_scale],
        )

        compare_trajectories_pointwise(
            impulse_loc_traj,
            py_traj.loc_traj[:sigma_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="sigma_auto_loc (LogNormal)",
        )

        impulse_final_scale = impulse_scale[1]
        numpyro_final_scale = py_traj.loc_traj[:sigma_auto_scale][end]
        @info "constrained final_scale: impulse=$impulse_final_scale numpyro=$numpyro_final_scale"
        @test isapprox(impulse_final_scale, numpyro_final_scale; atol=1e-12, rtol=1e-12)

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (LogNormal)",
        )
    end
end
