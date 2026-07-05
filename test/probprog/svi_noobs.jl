using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
const PNormal = ProbProg.Normal

include(joinpath(@__DIR__, "common.jl"))

# Expected ELBO at init (loc=0, σ=0.1): -2.303 + 0.495 eps².

function model(rng)
    _, x = ProbProg.sample(rng, PNormal(0.0, 1.0); symbol=:x)
    return x
end

function svi_program(
    rng, model, step_size, num_steps, selection, obs_tensor, constrained_addresses
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
    )
    return trace, elbo_traj, state.loc, state.scale, state.elbo
end

num_steps = 3
lr = 0.01
rng = ReactantRNG(ConcreteRArray(UInt64[0, 0]))
selection = ProbProg.select(ProbProg.Address(:x))
step_size = ConcreteRNumber(lr)
obs = ProbProg.Constraint()
constrained_addresses = ProbProg.extract_addresses(obs)
obs_tensor = Reactant.to_rarray(reshape(Float64[], 1, 0))

compiled, _ = ProbProg.with_trace() do
    @compile optimize = :probprog svi_program(
        rng, model, step_size, num_steps, selection, obs_tensor, constrained_addresses
    )
end
trace, elbo_traj, loc_final, scale_final, elbo_final = compiled(
    rng, model, step_size, num_steps, selection, obs_tensor, constrained_addresses
)

println("=== Impulse VI, no obs ===")
println("loc trajectory: ", Array(trace)[:, 1])
println("elbo trajectory: ", Array(elbo_traj))
println("Final loc/scale: $(Array(loc_final)) / $(Array(scale_final))")
