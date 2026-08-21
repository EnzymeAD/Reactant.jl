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
const T = 56
y_data = Float64[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]
_logfactorial(k) = k <= 1 ? 0.0 : sum(log, 2.0:1.0:Float64(k))
lg_data = Float64[_logfactorial(Int(v)) for v in y_data]

# Masked (rate-switch) encoding: one vector Poisson site, GenJAX-equivalent.
_y_sampler(rng, s, l1, l2, tvec, lg) = zero.(tvec)
function _y_logpdf(yv, s, l1, l2, tvec, lg)
    lam = ifelse.(tvec .< sum(s), sum(l1), sum(l2))
    return sum(yv .* log.(lam) .- lam .- lg)
end
function coal_masked(rng, tvec, lg)
    _, s = ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, Float64(T)); symbol=:s)
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    _, _ = ProbProg.sample(rng, _y_sampler, s, l1, l2, tvec, lg; symbol=:y, logpdf=_y_logpdf)
    return nothing
end
function mh_prog(rng, model, tvec, lg, bins, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(rng, obs_tensor, model, tvec, lg; constrained_addresses=constrained)
    hist = zero.(bins)
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec, lg;
            selection=ProbProg.select(ProbProg.Address(:s)))
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec, lg;
            selection=ProbProg.select(ProbProg.Address(:l1)))
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec, lg;
            selection=ProbProg.select(ProbProg.Address(:l2)))
        s_v = sum(trace[1:1, 1:1])
        hist = hist .+ ifelse.(abs.(bins .- s_v) .< 0.5, 1.0, 0.0)
    end
    return hist
end
tvec = Reactant.to_rarray(Float64.(0:(T - 1)))
lg = Reactant.to_rarray(lg_data)
obs = ProbProg.Constraint(:y => y_data, :l1 => [3.0], :l2 => [0.9])
constrained = ProbProg.extract_addresses(obs)
ot = ProbProg.flatten_constraint(obs)
bins = Reactant.to_rarray(reshape(Float64.(0:T), 1, T + 1))
NUM_SWEEPS = 2000
rng = ReactantRNG(Reactant.to_rarray(UInt64[12345, 0]))
ct = @elapsed begin
    global compiled, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog mh_prog(
            rng, coal_masked, tvec, lg, bins, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
    end
end
@printf("compile %.1fs\n", ct)
hist = Array(compiled(rng, coal_masked, tvec, lg, bins, ot, constrained, ConcreteRNumber(NUM_SWEEPS)))[:]
best = Inf
for _ in 1:3
    t = @elapsed begin
        global hist = Array(compiled(rng, coal_masked, tvec, lg, bins, ot, constrained, ConcreteRNumber(NUM_SWEEPS)))[:]
    end
    global best = min(best, t)
end
smap = argmax(hist) - 1
@printf("IMPULSE-MASKED coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))
