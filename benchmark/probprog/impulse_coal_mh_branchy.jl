using Reactant, Random, LinearAlgebra, Statistics, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
import Reactant.XLA.PJRT: Future
function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        while !isready(future); GC.safepoint(); yield(); end
    end
    return nothing
end
Reactant.set_default_backend("cpu")
include("/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog/standard/coal_mining.jl")
const T = 56
branch_model(rng) = coal_mining_branch(rng, T)
y = Float64[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]
mh_obs = branch_constraint(y)
mh_constrained = ProbProg.extract_addresses(mh_obs)
mh_ot = ProbProg.flatten_constraint(mh_obs)
bins = Reactant.to_rarray(reshape(Float64.(0:T), 1, T + 1))
NUM_SWEEPS = 2000
rng = ReactantRNG(Reactant.to_rarray(UInt64[12345, 0]))
ct = @elapsed begin
    global compiled, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog coal_mining_mh_program(
            rng, branch_model, bins, mh_ot, mh_constrained, ConcreteRNumber(NUM_SWEEPS))
    end
end
@printf("compile %.1fs\n", ct)
hist = Array(compiled(rng, branch_model, bins, mh_ot, mh_constrained, ConcreteRNumber(NUM_SWEEPS)))[:]
best = Inf
for _ in 1:3
    t = @elapsed begin
        global hist = Array(compiled(rng, branch_model, bins, mh_ot, mh_constrained, ConcreteRNumber(NUM_SWEEPS)))[:]
    end
    global best = min(best, t)
end
smap = argmax(hist) - 1
@printf("IMPULSE coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))
