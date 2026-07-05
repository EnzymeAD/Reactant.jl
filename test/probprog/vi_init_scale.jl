using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# initial_scale is in constrained (positive) space.
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
    init_scale,
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
        initial_position=init_loc,
        initial_scale=init_scale,
    )
    return trace, elbo_traj, state.loc, state.scale
end

@testset "vi_init_scale_pointwise" begin
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

    init_loc_val = 0.5
    init_scale_val = 0.3

    step_size = ConcreteRNumber(lr)

    init_loc = Reactant.to_rarray(reshape(Float64[init_loc_val], 1, 1))
    init_scale = Reactant.to_rarray(reshape(Float64[init_scale_val], 1, 1))
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
            init_scale,
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
        init_scale,
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
        numpyro_infer = pyimport("numpyro.infer")
        numpyro_optim = pyimport("numpyro.optim")
        autoguide_mod = pyimport("numpyro.infer.autoguide")
        np = pyimport("numpy")

        obs_y_py = obs_y
        numpyro_model = pyfunc(
            function ()
                x = numpyro_mod.sample("x", dist.Normal(0.0, 1.0))
                numpyro_mod.sample("y", dist.Normal(x, 0.5); obs=obs_y_py)
                return x
            end; name="model"
        )

        rng_key = seed_to_rbg_key(UInt64[0, 0])

        # NumPyro takes the constrained init scale through svi.init's init_params.
        guide = autoguide_mod.AutoNormal(numpyro_model)
        opt = numpyro_optim.Adam(; step_size=lr)
        elbo = numpyro_infer.Trace_ELBO(; num_particles=1)
        svi = numpyro_infer.SVI(numpyro_model, guide, opt; loss=elbo)

        init_params = pydict()
        init_params["x_auto_loc"] = jnp.array(Float64[init_loc_val])
        init_params["x_auto_scale"] = jnp.array(Float64[init_scale_val])
        state = svi.init(rng_key; init_params=init_params)

        loc_traj = Float64[]
        scale_traj = Float64[]
        elbo_traj = Float64[]
        for _ in 1:num_steps
            out = svi.update(state)
            state = out[0]
            loss = out[1]
            push!(elbo_traj, -pyconvert(Float64, np.asarray(loss).item()))
            params = svi.get_params(state)
            push!(
                loc_traj,
                pyconvert(Float64, np.asarray(params["x_auto_loc"]).flatten()[0].item()),
            )
            push!(
                scale_traj,
                pyconvert(Float64, np.asarray(params["x_auto_scale"]).flatten()[0].item()),
            )
        end

        compare_trajectories_pointwise(
            impulse_loc_traj,
            loc_traj;
            atol=1e-12,
            rtol=1e-12,
            label="x_auto_loc (initial_scale=$init_scale_val)",
        )

        impulse_final_scale = impulse_scale[1]
        numpyro_final_scale = scale_traj[end]
        @info "initial_scale: impulse_final_scale=$impulse_final_scale numpyro_final_scale=$numpyro_final_scale"
        @test isapprox(impulse_final_scale, numpyro_final_scale; atol=1e-12, rtol=1e-12)

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (initial_scale=$init_scale_val)",
        )
    end
end
