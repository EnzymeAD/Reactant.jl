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
const N = 30
const X = [0.0, 0.2759, 0.5517, 0.8276, 1.1034, 1.3793, 1.6552, 1.931, 2.2069, 2.4828, 2.7586, 3.0345, 3.3103, 3.5862, 3.8621, 4.1379, 4.4138, 4.6897, 4.9655, 5.2414, 5.5172, 5.7931, 6.069, 6.3448, 6.6207, 6.8966, 7.1724, 7.4483, 7.7241, 8.0]
const Y = [-0.4453, -0.8217, -0.4477, -0.3201, -0.9794, -0.1994, 0.5287, 0.1443, 1.2569, 1.8079, 1.0433, 1.9606, 2.2105, 3.1492, 2.9465, 3.6189, 2.5726, 3.3957, 3.6323, 4.1488, 5.7783, 4.2277, 5.3055, 5.5581, 5.9808, 5.735, 6.3052, 6.6012, 5.8466, 6.8494]

_unif_sampler(rng, shape) = Float64(2π) .* rand(rng, shape...)
_unif_logpdf(x, _) = -log(2π) .+ 0.0 .* sum(x)
_y_sampler(rng, mu) = zero.(mu)
_y_logpdf(yv, mu) = sum(.-log(0.5) .- 0.5 * log(2π) .- (yv .- mu) .^ 2 ./ (2 * 0.5^2))

function linesine_model(rng, xs)
    _, m = ProbProg.sample(rng, ProbProg.Bernoulli(0.0, (1,)); symbol=:m)
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, intercept = ProbProg.sample(rng, ProbProg.Normal(0.0, 3.0, (1,)); symbol=:intercept)
    _, amp = ProbProg.sample(rng, ProbProg.Normal(0.0, 3.0, (1,)); symbol=:amp)
    _, phase = ProbProg.sample(rng, _unif_sampler, (1,); symbol=:phase, logpdf=_unif_logpdf)
    mval = sum(m)
    muL = sum(slope) .* xs .+ sum(intercept)
    muS = sum(amp) .* sin.(Float64(2π) .* xs ./ 4.0 .+ sum(phase))
    mu = mval .* muL .+ (1.0 - mval) .* muS
    _, _ = ProbProg.sample(rng, _y_sampler, mu; symbol=:y, logpdf=_y_logpdf)
    return nothing
end

function switch_prog(rng, model, xs, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(rng, obs_tensor, model, xs; constrained_addresses=constrained)
    macc = weight .* 0.0
    @trace for _ in 1:num_iters
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:m)))
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:slope)))
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)))
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:amp)))
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:phase)))
        macc = macc .+ sum(trace[1:1, 1:1])
    end
    return trace, weight, macc
end

xs = Reactant.to_rarray(X)
obs = ProbProg.Constraint(:y => Y)
constrained = ProbProg.extract_addresses(obs)
ot = ProbProg.flatten_constraint(obs)
NUM_SWEEPS = 1000
MOVES = NUM_SWEEPS * 5
rng = ReactantRNG(Reactant.to_rarray(UInt64[4242, 0]))
ct = @elapsed begin
    global compiled, tt = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog switch_prog(
            rng, linesine_model, xs, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
    end
end
@printf("compile %.1fs\n", ct)
out = compiled(rng, linesine_model, xs, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
best = Inf
for _ in 1:20
    t = @elapsed begin
        global out = compiled(rng, linesine_model, xs, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
        Array(out[1])
    end
    global best = min(best, t)
    @printf("IMPULSE linesine REP wall %.6f s | per-move %.4f us\n", t, t / MOVES * 1e6)
end
pline = Float64(out[3]) / NUM_SWEEPS
@printf("IMPULSE linesine per-move %.3f us | P(m=line) = %.4f\n", best / MOVES * 1e6, pline)
