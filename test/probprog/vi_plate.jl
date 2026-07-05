using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Normal(mu, sigma, (N,)) matches NumPyro's plate: one rng draw of N values.
const NDATA = 4
const DATA = Float64[1.0, 0.5, -0.3, 1.2]

function model(rng)
    _, mu = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:mu)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(mu, 0.5, (NDATA,)); symbol=:y)
    return mu
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

@testset "vi_plate_pointwise" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    # Constraining every site keeps generate from consuming rng (NumPyro parity).
    obs = ProbProg.Constraint(:mu => [0.0], :y => DATA)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:mu))
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
        # NumPyro plate model needs to be defined in Python (with-block syntax).
        py = pyimport("builtins")
        sys = pyimport("sys")
        builtins = pyimport("builtins")
        module_src = """
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

NDATA = $NDATA
DATA = jnp.array($(repr(DATA)))

def plate_model():
    mu = numpyro.sample(\"mu\", dist.Normal(0.0, 1.0))
    with numpyro.plate(\"data\", NDATA):
        numpyro.sample(\"y\", dist.Normal(mu, 0.5), obs=DATA)
    return mu
"""
        ns = pydict()
        builtins.exec(module_src, ns)
        numpyro_model = ns["plate_model"]

        rng_key = seed_to_rbg_key(UInt64[0, 0])
        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            init_loc_values=Dict(:mu => 0.0),
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:mu_auto_loc, :mu_auto_scale],
        )

        compare_trajectories_pointwise(
            impulse_loc_traj,
            py_traj.loc_traj[:mu_auto_loc];
            atol=1e-12,
            rtol=1e-12,
            label="mu_auto_loc (plate, N=$NDATA)",
        )

        impulse_final_scale = impulse_scale[1]
        numpyro_final_scale = py_traj.loc_traj[:mu_auto_scale][end]
        @info "plate final_scale: impulse=$impulse_final_scale numpyro=$numpyro_final_scale"
        @test isapprox(impulse_final_scale, numpyro_final_scale; atol=1e-12, rtol=1e-12)

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-12,
            rtol=1e-12,
            label="elbo_traj (plate, N=$NDATA)",
        )
    end
end
