# Line vs sine with a sampled period (2026-10-03, evidence
# oopsla/evidence/fig13_stockwait_2026-10-03): the model of Gen's "Introduction
# to Modeling in Gen" samples the sine's period, period ~ gamma(5, 1) (Gen's
# gamma(shape, scale): shape 5, scale 1), and the sine mean is
# amp * sin(2 pi x / period + phase). Otherwise identical to gen_linesine_mh_harness.jl
# (priors, data, seed); the MH sweep gains a prior-resimulation move on period
# (six moves; per-move = wall / (6 * sweeps)). generate draws every latent,
# period included, from its prior (only ys is constrained). Mean construct: both means computed, then m ? muL : muS.
using Gen, Random, Printf
const N = 30
const X = [0.0, 0.2759, 0.5517, 0.8276, 1.1034, 1.3793, 1.6552, 1.931, 2.2069, 2.4828, 2.7586, 3.0345, 3.3103, 3.5862, 3.8621, 4.1379, 4.4138, 4.6897, 4.9655, 5.2414, 5.5172, 5.7931, 6.069, 6.3448, 6.6207, 6.8966, 7.1724, 7.4483, 7.7241, 8.0]
const Y = [-0.4453, -0.8217, -0.4477, -0.3201, -0.9794, -0.1994, 0.5287, 0.1443, 1.2569, 1.8079, 1.0433, 1.9606, 2.2105, 3.1492, 2.9465, 3.6189, 2.5726, 3.3957, 3.6323, 4.1488, 5.7783, 4.2277, 5.3055, 5.5581, 5.9808, 5.735, 6.3052, 6.6012, 5.8466, 6.8494]
@gen function linesine(xs)
    m ~ bernoulli(0.5)
    slope ~ normal(0.0, 2.0)
    intercept ~ normal(0.0, 3.0)
    amp ~ normal(0.0, 3.0)
    phase ~ uniform(0.0, 2π)
    period ~ gamma(5.0, 1.0)
    muL = slope .* xs .+ intercept
    muS = amp .* sin.(2π .* xs ./ period .+ phase)
    mu = m ? muL : muS
    ys ~ broadcasted_normal(mu, fill(0.5, length(xs)))
end
obs = choicemap((:ys, Y))
const NUM_SWEEPS = 1000
const MOVES = NUM_SWEEPS * 6
function kern()
    tr, _ = generate(linesine, (X,), obs)
    macc = 0
    for _ in 1:NUM_SWEEPS
        tr, _ = mh(tr, select(:m))
        tr, _ = mh(tr, select(:slope))
        tr, _ = mh(tr, select(:intercept))
        tr, _ = mh(tr, select(:amp))
        tr, _ = mh(tr, select(:phase))
        tr, _ = mh(tr, select(:period))
        macc += tr[:m] ? 1 : 0
    end
    return macc
end
Random.seed!(4242)
macc = kern()
best = Inf
for _ in 1:20
    t = @elapsed begin
        global macc = kern()
    end
    global best = min(best, t)
    @printf("GENJL linesine_period REP wall %.6f s | per-move %.4f us\n", t, t / MOVES * 1e6)
end
@printf("GENJL linesine_period per-move %.3f us | P(m=line) = %.4f\n", best / MOVES * 1e6, macc / NUM_SWEEPS)
