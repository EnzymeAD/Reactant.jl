using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Data: y_obs and sigma_obs from Rubin (1981).
const J = 8
const Y_OBS = Float64[28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const SIGMA_OBS = Float64[15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

function model(rng, sigma)
    _, mu = ProbProg.sample(rng, ProbProg.Normal(0.0, 5.0); symbol=:mu)
    _, tau = ProbProg.sample(rng, ProbProg.LogNormal(0.0, 1.0); symbol=:tau)
    _, theta = ProbProg.sample(rng, ProbProg.Normal(mu, tau, (J,)); symbol=:theta)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(theta, sigma, (J,)); symbol=:y)
    return (mu, tau, theta)
end

function svi_program(
    rng,
    model,
    sigma,
    step_size,
    num_steps::Int,
    selection,
    obs_tensor,
    constrained_addresses,
    init_loc,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model, sigma; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng,
        trace0,
        model,
        sigma;
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

@testset "vi_eight_schools" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    # Constraining every site keeps generate from consuming rng (NumPyro parity).
    obs = ProbProg.Constraint(
        :mu => [0.0], :tau => [1.0], :theta => zeros(Float64, J), :y => Y_OBS
    )
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(
        ProbProg.Address(:mu), ProbProg.Address(:tau), ProbProg.Address(:theta)
    )
    sigma = Reactant.to_rarray(SIGMA_OBS)
    lr = 0.01
    num_steps = 5
    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, J + 2))

    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_program(
            rng,
            model,
            sigma,
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
        sigma,
        step_size,
        num_steps,
        selection,
        obs_tensor,
        constrained_addresses,
        init_loc,
    )

    impulse_loc_traj = Array(impulse_trace)
    impulse_elbo_traj = Array(impulse_elbo)
    impulse_loc_final = Array(impulse_loc)
    impulse_scale_final = Array(impulse_scale)

    @test size(impulse_loc_traj) == (num_steps, J + 2)
    @test size(impulse_elbo_traj) == (num_steps,)

    if check_numpyro_available()
        builtins = pyimport("builtins")
        module_src = """
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

J = $J
Y_OBS = jnp.array($(repr(Y_OBS)))
SIGMA_OBS = jnp.array($(repr(SIGMA_OBS)))

def eight_schools():
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.LogNormal(0.0, 1.0))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("y", dist.Normal(theta, SIGMA_OBS), obs=Y_OBS)
    return mu, tau, theta
"""
        ns = pydict()
        builtins.exec(module_src, ns)
        numpyro_model = ns["eight_schools"]

        rng_key = seed_to_rbg_key(UInt64[0, 0])
        jnp = pyimport("jax.numpy")
        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            init_loc_values=Dict(:mu => 0.0, :tau => 1.0, :theta => jnp.zeros(J)),
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[
                :mu_auto_loc,
                :tau_auto_loc,
                :theta_auto_loc,
                :mu_auto_scale,
                :tau_auto_scale,
                :theta_auto_scale,
            ],
        )

        # Position layout: [mu, tau, theta[1..J]]
        impulse_mu_traj = impulse_loc_traj[:, 1]
        compare_trajectories_pointwise(
            reshape(impulse_mu_traj, :, 1),
            py_traj.loc_traj[:mu_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="mu_auto_loc (8 schools)",
        )

        # tau is not compared: NumPyro's tau_auto_loc is constrained, ours is log-space.

        @info "8 schools elbo iter 1: impulse=$(impulse_elbo_traj[1]) numpyro=$(py_traj.elbo_traj[1])"
        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-10,
            rtol=1e-10,
            label="elbo_traj (8 schools)",
        )
    end
end
