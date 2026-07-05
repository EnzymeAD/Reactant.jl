using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

const D = 3
const N = 20
const SEED_DATA = 42

function gen_data()
    rng = Random.MersenneTwister(SEED_DATA)
    X = randn(rng, N, D)
    w_true = Float64[0.5, -1.0, 1.5]
    sigma_true = 0.3
    y = X * w_true .+ sigma_true * randn(rng, N)
    return X, y
end
const X_DATA, Y_DATA = gen_data()

function model(rng, X)
    _, w = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (D,)); symbol=:w)
    _, sigma = ProbProg.sample(rng, ProbProg.LogNormal(0.0, 1.0); symbol=:sigma)
    yhat = X * w
    _, _y = ProbProg.sample(rng, ProbProg.Normal(yhat, sigma, (N,)); symbol=:y)
    return (w, sigma)
end

function svi_program(
    rng,
    model,
    X,
    step_size,
    num_steps::Int,
    selection,
    obs_tensor,
    constrained_addresses,
    init_loc,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model, X; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng,
        trace0,
        model,
        X;
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

@testset "vi_bayesian_linreg" begin
    seed = Reactant.to_rarray(UInt64[0, 0])
    rng = ReactantRNG(seed)

    obs = ProbProg.Constraint(:w => zeros(Float64, D), :sigma => [1.0], :y => Y_DATA)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:w), ProbProg.Address(:sigma))
    X = Reactant.to_rarray(X_DATA)
    lr = 0.01
    num_steps = 5
    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, D + 1))

    compiled, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_program(
            rng,
            model,
            X,
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
        X,
        step_size,
        num_steps,
        selection,
        obs_tensor,
        constrained_addresses,
        init_loc,
    )

    impulse_loc_traj = Array(impulse_trace)
    impulse_elbo_traj = Array(impulse_elbo)

    @test size(impulse_loc_traj) == (num_steps, D + 1)

    if check_numpyro_available()
        builtins = pyimport("builtins")
        x_repr =
            "[" *
            join(["[" * join(string.(X_DATA[i, :]), ", ") * "]" for i in 1:N], ", ") *
            "]"
        y_repr = "[" * join(string.(Y_DATA), ", ") * "]"
        module_src = """
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

D = $D
N = $N
X_DATA = jnp.array($x_repr)
Y_DATA = jnp.array($y_repr)

def linreg_model():
    w = numpyro.sample(\"w\", dist.Normal(jnp.zeros(D), jnp.ones(D)).to_event(1))
    sigma = numpyro.sample(\"sigma\", dist.LogNormal(0.0, 1.0))
    yhat = X_DATA @ w
    with numpyro.plate(\"data\", N):
        numpyro.sample(\"y\", dist.Normal(yhat, sigma), obs=Y_DATA)
    return w, sigma
"""
        ns = pydict()
        builtins.exec(module_src, ns)
        numpyro_model = ns["linreg_model"]

        rng_key = seed_to_rbg_key(UInt64[0, 0])
        jnp = pyimport("jax.numpy")
        py_traj = run_numpyro_svi(;
            model=numpyro_model,
            rng_key=rng_key,
            init_loc_values=Dict(:w => jnp.zeros(D), :sigma => 1.0),
            init_scale=0.1,
            num_steps=num_steps,
            step_size=lr,
            optimizer=:Adam,
            num_particles=1,
            param_names=[:w_auto_loc, :sigma_auto_loc, :w_auto_scale, :sigma_auto_scale],
        )

        impulse_w_traj_first = impulse_loc_traj[:, 1]
        @info "linreg w[1] iter 1: impulse=$(impulse_w_traj_first[1]) numpyro=$(py_traj.loc_traj[:w_auto_loc][1])"
        @info "linreg elbo iter 1: impulse=$(impulse_elbo_traj[1]) numpyro=$(py_traj.elbo_traj[1])"

        compare_trajectories_pointwise(
            reshape(impulse_w_traj_first, :, 1),
            py_traj.loc_traj[:w_auto_loc];
            atol=1e-10,
            rtol=1e-10,
            label="w[1] (Bayesian linreg, D=$D, N=$N)",
        )

        compare_trajectories_pointwise(
            reshape(impulse_elbo_traj, :, 1),
            py_traj.elbo_traj;
            atol=1e-10,
            rtol=1e-10,
            label="elbo_traj (Bayesian linreg, D=$D, N=$N)",
        )
    end
end
