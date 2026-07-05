using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

const K = 4
const Y_DATA = Float64[1.0, 0.5, -0.3, 1.2]

function model(rng)
    _, mu = ProbProg.sample(rng, ProbProg.Normal(0.0, 5.0); symbol=:mu)
    _, theta = ProbProg.sample(rng, ProbProg.Normal(mu, 1.0, (K,)); symbol=:theta)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(theta, 0.5, (K,)); symbol=:y)
    return (mu, theta)
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

@testset "vi_plate_hierarchical" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs = ProbProg.Constraint(:mu => [0.0], :theta => zeros(Float64, K), :y => Y_DATA)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:mu), ProbProg.Address(:theta))
    lr = 0.01
    num_steps = 5
    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, K + 1))

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
    impulse_loc_final = Array(impulse_loc)
    impulse_scale_final = Array(impulse_scale)

    @test size(impulse_loc_traj) == (num_steps, K + 1)
    @test size(impulse_elbo_traj) == (num_steps,)

    if check_numpyro_available()
        builtins = pyimport("builtins")
        module_src = """
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

K = $K
DATA = jnp.array($(repr(Y_DATA)))

def hier_model():
    mu = numpyro.sample(\"mu\", dist.Normal(0.0, 5.0))
    with numpyro.plate(\"groups\", K):
        theta = numpyro.sample(\"theta\", dist.Normal(mu, 1.0))
        numpyro.sample(\"y\", dist.Normal(theta, 0.5), obs=DATA)
    return mu, theta
"""
        ns = pydict()
        builtins.exec(module_src, ns)
        numpyro_model = ns["hier_model"]

        rng_key = seed_to_rbg_key(UInt64[0, 0])
        jnp = pyimport("jax.numpy")
        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            init_loc_values=Dict(:mu => 0.0, :theta => jnp.zeros(K)),
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:mu_auto_loc, :theta_auto_loc, :mu_auto_scale, :theta_auto_scale],
        )

        # Position layout [mu, theta[1..K]]; only mu is compared.
        impulse_mu_traj = impulse_loc_traj[:, 1]
        compare_trajectories_pointwise(
            reshape(impulse_mu_traj, :, 1),
            py_traj.loc_traj[:mu_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="mu_auto_loc (hierarchical plate, K=$K)",
        )

        @info "elbo iter 1: impulse=$(impulse_elbo_traj[1]) numpyro=$(py_traj.elbo_traj[1])"
        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (hierarchical plate, K=$K)",
        )
    end
end
