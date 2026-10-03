# Line vs sine with a sampled period, BRANCH encoding (2026-10-03, evidence
# oopsla/evidence/fig13_stockwait_2026-10-03). The model of Gen's "Introduction
# to Modeling in Gen" samples the sine's period, period ~ gamma(5, 1) (shape 5,
# scale 1), and the sine mean is amp * sin(2 pi x / period + phase); every other
# prior is as in impulse_linesine_mh_harness.jl. The mean is selected by
# ProbProg.if_ on m (a deterministic branch, no sample site inside), the way the
# tutorial writes `if is_line ... else ... end`; the arms share no expression.
# The MH sweep is six prior-resimulation moves (m, slope, intercept, amp, phase,
# period); per-move = wall / (6 * sweeps).
#
# Start: generate cannot draw an unconstrained Gamma site in Impulse (the Gamma
# sampler is a traced rejection loop that only the simulate and regenerate
# lowerings call), so the program draws the starting period itself with that
# sampler (ProbProg._gamma_sampler) and splices it into the constraint at a
# sentinel slot, as impulse_coal_mh_masked.jl does for its rates; generate draws
# every other latent from its prior, and the MH move on period regenerates it
# from the prior. Every timed repetition starts from a fresh prior draw.
using Reactant, Random, LinearAlgebra, Statistics, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
# Stock Base.wait (blocks, like JAX's block_until_ready): a spin-yield override competed with the PJRT worker thread on one pinned CPU.
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
    _, period = ProbProg.sample(rng, ProbProg.Gamma(5.0, 1.0, (1,)); symbol=:period)
    mu = ProbProg.if_(rng, sum(m) > 0.5,
        () -> sum(slope) .* xs .+ sum(intercept),
        () -> sum(amp) .* sin.(Float64(2π) .* xs ./ sum(period) .+ sum(phase)))
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
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs;
            selection=ProbProg.select(ProbProg.Address(:period)))
        macc = macc .+ sum(trace[1:1, 1:1])
    end
    return trace, weight, macc
end

xs = Reactant.to_rarray(X)
# The period slot of the flattened constraint is located with a sentinel value.
const PERIOD_SENTINEL = -12345.0
obs = ProbProg.Constraint(:y => Y, :period => [PERIOD_SENTINEL])
constrained = ProbProg.extract_addresses(obs)
ot = ProbProg.flatten_constraint(obs)
_flat = vec(Array(ot))
_ip = findfirst(==(PERIOD_SENTINEL), _flat)
_onehot(i) = (v = zeros(1, length(_flat)); v[1, i] = 1.0; v)
keep = Reactant.to_rarray(1.0 .- _onehot(_ip))
ep = Reactant.to_rarray(_onehot(_ip))
function prior_start(rng, ot, keep, ep)
    g = ProbProg._gamma_sampler(rng, 5.0, 1.0, (1,))
    return ot .* keep .+ ep .* sum(g)
end
switch_prog_prior(rng, model, xs, ot, keep, ep, constrained, num_iters) =
    switch_prog(rng, model, xs, prior_start(rng, ot, keep, ep), constrained, num_iters)
NUM_SWEEPS = 1000
MOVES = NUM_SWEEPS * 6
rng = ReactantRNG(Reactant.to_rarray(UInt64[4242, 0]))
args = (rng, linesine_model, xs, ot, keep, ep, constrained, ConcreteRNumber(NUM_SWEEPS))
ct = @elapsed begin
    global compiled, tt = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog switch_prog_prior(args...)
    end
end
@printf("compile %.1fs\n", ct)
out = compiled(args...)
best = Inf
for _ in 1:20
    t = @elapsed begin
        global out = compiled(args...)
        Array(out[1])
    end
    global best = min(best, t)
    @printf("IMPULSE linesine_period_branch REP wall %.6f s | per-move %.4f us\n", t, t / MOVES * 1e6)
end
pline = Float64(out[3]) / NUM_SWEEPS
@printf("IMPULSE linesine_period_branch per-move %.3f us | P(m=line) = %.4f\n", best / MOVES * 1e6, pline)
