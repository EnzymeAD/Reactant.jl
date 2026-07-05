using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# No init on either side: both fall back to init_to_uniform(radius=2).
function model(rng)
    _, x = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:x)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(x, 0.5); symbol=:y)
    return x
end

function svi_program(
    rng, model, step_size, num_steps::Int, selection, obs_tensor, constrained_addresses
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
        init_radius=2.0,
    )
    return trace, elbo_traj, state.loc, state.scale
end

@testset "vi_init_to_uniform_pointwise" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs_y = 1.0
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

    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_program(
            rng,
            model,
            step_size,
            num_steps,
            selection,
            obs_tensor,
            constrained_addresses,
        )
    end
    impulse_trace, impulse_elbo, _, impulse_scale = compiled(
        rng, model, step_size, num_steps, selection, obs_tensor, constrained_addresses
    )
    impulse_loc_traj = Array(impulse_trace)
    impulse_elbo_traj = Array(impulse_elbo)

    @test size(impulse_loc_traj) == (num_steps, 1)

    if check_numpyro_available()
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
            init_loc_values=nothing,
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:x_auto_loc, :x_auto_scale],
        )

        compare_trajectories_pointwise(
            impulse_loc_traj,
            py_traj.loc_traj[:x_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="x_auto_loc (init_to_uniform)",
        )

        impulse_final_scale = impulse_scale[1]
        numpyro_final_scale = py_traj.loc_traj[:x_auto_scale][end]
        @info "init_to_uniform final_scale: impulse=$impulse_final_scale numpyro=$numpyro_final_scale"
        @test isapprox(impulse_final_scale, numpyro_final_scale; atol=1e-12, rtol=1e-12)

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (init_to_uniform)",
        )
    end
end
