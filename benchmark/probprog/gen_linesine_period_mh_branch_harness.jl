# Line vs sine with a sampled period (2026-10-03, evidence
# oopsla/evidence/fig13_stockwait_2026-10-03): the model of Gen's "Introduction
# to Modeling in Gen" samples the sine's period, period ~ gamma(5, 1) (Gen's
# gamma(shape, scale): shape 5, scale 1), and the sine mean is
# amp * sin(2 pi x / period + phase). Otherwise identical to gen_linesine_mh_branch_harness.jl
# (priors, data, seed); the MH sweep gains a prior-resimulation move on period
# (six moves; per-move = wall / (6 * sweeps)). generate draws every latent,
# period included, from its prior (only ys is constrained). Mean construct: plain `if m ... else ... end` (the tutorial's branch).
using Gen, Random, Printf
const N = 30
const X = [0.0, 0.2759, 0.5517, 0.8276, 1.1034, 1.3793, 1.6552, 1.931, 2.2069, 2.4828, 2.7586, 3.0345, 3.3103, 3.5862, 3.8621, 4.1379, 4.4138, 4.6897, 4.9655, 5.2414, 5.5172, 5.7931, 6.069, 6.3448, 6.6207, 6.8966, 7.1724, 7.4483, 7.7241, 8.0]
const Y = [-0.4453, -0.8217, -0.4477, -0.3201, -0.9794, -0.1994, 0.5287, 0.1443, 1.2569, 1.8079, 1.0433, 1.9606, 2.2105, 3.1492, 2.9465, 3.6189, 2.5726, 3.3957, 3.6323, 4.1488, 5.7783, 4.2277, 5.3055, 5.5581, 5.9808, 5.735, 6.3052, 6.6012, 5.8466, 6.8494]
# GEN_LS_LIK (2026-10-03, figure 13 round 2):
#   broadcast  (default) ys ~ broadcasted_normal(mu, fill(0.5, N)), as through the
#              2026-10-03 single window: a log of the std per datum in every move
#   fused      ys ~ fused_normal(mu, 0.5), the user-defined vector normal with a
#              scalar std of gen_mh_linreg_tutorial_harness.jl (vec_custom): one
#              log per move, the per-move terms of the Impulse and GenJAX
#              encodings of this row (their scalar 0.5 is a compile-time constant)
const LIK = get(ENV, "GEN_LS_LIK", "broadcast")
LIK in ("broadcast", "fused") || error("GEN_LS_LIK must be broadcast or fused")
const FUSED = LIK == "fused"   # a constant Bool, so the branch below costs nothing per move
struct FusedNormalScalar <: Gen.Distribution{Vector{Float64}} end
const fused_normal = FusedNormalScalar()
function Gen.random(::FusedNormalScalar, mu::Vector{Float64}, sig::Float64)
    return mu .+ sig .* randn(length(mu))
end
function Gen.logpdf(::FusedNormalScalar, y::Vector{Float64}, mu::Vector{Float64}, sig::Float64)
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        d = y[i] - mu[i]
        s += d * d
    end
    n = length(y)
    return -s / (2 * sig^2) - n * log(sig) - n / 2 * log(2π)
end
Gen.logpdf_grad(::FusedNormalScalar, y, mu, sig) = (nothing, nothing, nothing)
Gen.has_output_grad(::FusedNormalScalar) = false
Gen.has_argument_grads(::FusedNormalScalar) = (false, false)
Gen.is_discrete(::FusedNormalScalar) = false
(::FusedNormalScalar)(mu, sig) = Gen.random(fused_normal, mu, sig)
@gen function linesine(xs)
    m ~ bernoulli(0.5)
    slope ~ normal(0.0, 2.0)
    intercept ~ normal(0.0, 3.0)
    amp ~ normal(0.0, 3.0)
    phase ~ uniform(0.0, 2π)
    period ~ gamma(5.0, 1.0)
    if m
        mu = slope .* xs .+ intercept
    else
        mu = amp .* sin.(2π .* xs ./ period .+ phase)
    end
    if FUSED
        ys ~ fused_normal(mu, 0.5)
    else
        ys ~ broadcasted_normal(mu, fill(0.5, length(xs)))
    end
end
obs = choicemap((:ys, Y))
const NUM_SWEEPS = 1000
const MOVES = NUM_SWEEPS * 6
const LABEL = "GENJL linesine_period_branch" * (FUSED ? "_fused" : "")
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
    @printf("%s REP wall %.6f s | per-move %.4f us\n", LABEL, t, t / MOVES * 1e6)
end
@printf("%s per-move %.3f us | P(m=line) = %.4f\n", LABEL, best / MOVES * 1e6, macc / NUM_SWEEPS)
