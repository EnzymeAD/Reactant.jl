using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Two sites so AutoMultivariateNormal is non-degenerate.
function model(rng)
    _, x1 = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:x1)
    _, x2 = ProbProg.sample(rng, ProbProg.Normal(x1, 1.0); symbol=:x2)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(x1 + x2, 0.5); symbol=:y)
    return (x1, x2)
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
        guide=:AutoMultivariateNormal,
        optimizer=:Adam,
        step_size,
        num_steps,
        num_particles=1,
        init_scale=0.1,
        initial_position=init_loc,
    )
    return trace, elbo_traj, state.loc, state.scale
end

@testset "vi_mvn_pointwise" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs_y = 1.0
    obs = ProbProg.Constraint(:x1 => [0.0], :x2 => [0.0], :y => [obs_y])
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:x1), ProbProg.Address(:x2))
    lr = 0.01
    num_steps = 5
    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, 2))

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
    impulse_trace, impulse_elbo, _, impulse_scale = compiled(
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
    impulse_scale_tril = Array(impulse_scale)

    @test size(impulse_loc_traj) == (num_steps, 2)
    @test size(impulse_scale_tril) == (2, 2)

    if check_numpyro_available()
        random_mod = pyimport("jax.random")
        numpyro_mod = pyimport("numpyro")
        dist = pyimport("numpyro.distributions")
        autoguide_mod = pyimport("numpyro.infer.autoguide")
        numpyro_optim = pyimport("numpyro.optim")
        numpyro_infer = pyimport("numpyro.infer")
        numpyro_init = pyimport("numpyro.infer.initialization")
        np = pyimport("numpy")

        obs_y_py = obs_y
        numpyro_model = pyfunc(
            function ()
                x1 = numpyro_mod.sample("x1", dist.Normal(0.0, 1.0))
                x2 = numpyro_mod.sample("x2", dist.Normal(x1, 1.0))
                numpyro_mod.sample("y", dist.Normal(x1 + x2, 0.5); obs=obs_y_py)
                return (x1, x2)
            end;
            name="model",
        )

        rng_key = seed_to_rbg_key(UInt64[0, 0])
        # Pin both sites' constrained init to 0 so unconstrained init is 0.
        init_loc_fn = numpyro_init.init_to_value(; values=pydict(; x1=0.0, x2=0.0))
        guide = autoguide_mod.AutoMultivariateNormal(
            numpyro_model; init_loc_fn=init_loc_fn, init_scale=0.1
        )
        opt = numpyro_optim.Adam(; step_size=lr)
        elbo = numpyro_infer.Trace_ELBO(; num_particles=1)
        svi = numpyro_infer.SVI(numpyro_model, guide, opt; loss=elbo)

        state = svi.init(rng_key)
        loc_traj = zeros(Float64, num_steps, 2)
        elbo_traj = zeros(Float64, num_steps)
        for step in 1:num_steps
            out = svi.update(state)
            state = out[0]
            loss = out[1]
            params = svi.get_params(state)
            loc_arr = np.asarray(params["auto_loc"]).flatten()
            for j in 1:2
                loc_traj[step, j] = pyconvert(Float64, loc_arr[j - 1].item())
            end
            elbo_traj[step] = -pyconvert(Float64, np.asarray(loss).item())
        end
        params_final = svi.get_params(state)
        scale_tril_np = np.asarray(params_final["auto_scale_tril"])
        numpyro_scale_tril = zeros(Float64, 2, 2)
        for i in 1:2, j in 1:2
            numpyro_scale_tril[i, j] = pyconvert(
                Float64, scale_tril_np[i - 1, j - 1].item()
            )
        end

        compare_trajectories_pointwise(
            impulse_loc_traj,
            vec(loc_traj);
            atol=1e-12,
            rtol=1e-12,
            label="auto_loc trajectory (MVN)",
        )

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo trajectory (MVN)",
        )

        compare_trajectories_pointwise(
            impulse_scale_tril,
            vec(numpyro_scale_tril);
            atol=1e-12,
            rtol=1e-12,
            label="final scale_tril (MVN)",
        )
    end
end
