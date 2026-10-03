# Gen.jl cell of the Gen-tutorial line-model row of the generality figure.
#
# Model: `line_model` of Gen's "Introduction to Modeling in Gen"
# (https://www.gen.dev/tutorials/intro-to-modeling/tutorial). Kernel: the block
# prior-resimulation move of "Iterative Inference in Gen"
# (https://www.gen.dev/tutorials/iterative-inference/tutorial), here
# mh(tr, select(:slope, :intercept)) once per sweep, 500 sweeps, after generate
# with the ys observed. Timed span: generate + loop, 20 repetitions after one
# discarded warm-up; per-move = wall / 500.
#
# Encodings (all measured; the figure takes the best one, as for the other
# figure-13 rows):
#   tutorial     the tutorial's model verbatim: one address (:y, i) per point
#   vec_dynamic  one vector site ys ~ broadcasted_normal (the encoding of
#                gen_mh_harness.jl's linreg_dynamic)
#   vec_static   the same in the static modeling language
#   vec_custom   one vector site ys ~ fused_normal(mu, 0.1): a user-defined
#                vector-normal distribution with a scalar std and a one-pass
#                logpdf, -N log s - N/2 log 2pi - sum((y - mu)^2) / (2 s^2), the
#                density formula of the Impulse harness's normal_logpdf, in the
#                style of the coal row's custom VecPoisson (2026-10-03,
#                FIX_LEDGER C2; Gen's broadcasted_normal takes the log of the
#                std per point even for a scalar std, 42 of about 60 us per move
#                at N=10000)
#   vec_custom_static  the same in the static modeling language
# After the timed repetitions (untimed): the trace score of a fixed trace against
# the hand-computed joint log density, acceptance count, final slope/intercept,
# and a record run (Random.seed!(1), the warm-up's chain) that prints every
# accepted sweep and the final state exactly, for comparison between encodings.
#
# Usage: julia --project=gen_env gen_mh_linreg_tutorial_harness.jl <data.json> <A|B>
using Gen, Random, Printf, Statistics

const DATA = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "linreg_tutorial_data.json")
const ROW = length(ARGS) >= 2 ? ARGS[2] : "A"
const ENCODINGS = split(get(ENV, "GEN_LT_ENCODINGS", "tutorial vec_dynamic vec_static"))
const NUM_SWEEPS = 500
const NREPS = 20

function read_row(path, row)
    s = read(path, String)
    i = findfirst("\"$row\": {", s)
    i === nothing && error("row $row not in $path")
    seg = s[last(i):end]
    vec(key) = begin
        j = findfirst("\"$key\": [", seg)
        k = findnext(']', seg, last(j))
        parse.(Float64, split(seg[(last(j) + 1):(k - 1)], ","))
    end
    return vec("xs"), vec("ys")
end
xs, ys = read_row(DATA, ROW)
const N = length(xs)

# Verbatim from the tutorial (the returned closure y is kept).
@gen function line_model(xs::Vector{Float64})
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))

    function y(x)
        return slope * x + intercept
    end

    for (i, x) in enumerate(xs)
        ({(:y, i)} ~ normal(y(x), 0.1))
    end

    return y
end

@gen function line_model_vec(xs::Vector{Float64}, sig::Vector{Float64})
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    ys ~ broadcasted_normal(slope .* xs .+ intercept, sig)
    return ys
end

line_mu(slope, intercept, xs) = slope .* xs .+ intercept
@gen (static) function line_model_vec_static(xs::Vector{Float64}, sig::Vector{Float64})
    slope ~ normal(0, 1)
    intercept ~ normal(0, 2)
    mu = line_mu(slope, intercept, xs)
    ys ~ broadcasted_normal(mu, sig)
    return ys
end

# Custom distribution of vec_custom(_static), as in the checker's prototype
# (oopsla/evidence/speedup_prediction_2026-10-03/fig13_generality/scripts/genjl/gen_models.jl).
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

@gen function line_model_custom(xs::Vector{Float64}, sig::Float64)
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    ys ~ fused_normal(slope .* xs .+ intercept, sig)
    return ys
end
@gen (static) function line_model_custom_static(xs::Vector{Float64}, sig::Float64)
    slope ~ normal(0, 1)
    intercept ~ normal(0, 2)
    mu = line_mu(slope, intercept, xs)
    ys ~ fused_normal(mu, sig)
    return ys
end

sig = fill(0.1, N)
function setup(enc)
    if enc == "tutorial"
        obs = choicemap()
        for (i, y) in enumerate(ys)
            obs[(:y, i)] = y
        end
        return line_model, (xs,), obs
    elseif enc == "vec_dynamic"
        return line_model_vec, (xs, sig), choicemap((:ys, ys))
    elseif enc == "vec_static"
        return line_model_vec_static, (xs, sig), choicemap((:ys, ys))
    elseif enc == "vec_custom"
        return line_model_custom, (xs, 0.1), choicemap((:ys, ys))
    elseif enc == "vec_custom_static"
        return line_model_custom_static, (xs, 0.1), choicemap((:ys, ys))
    end
    return error("unknown encoding $enc")
end

function kernel(model, args, obs, num_sweeps)
    tr, _ = generate(model, args, obs)
    acc = 0
    block = select(:slope, :intercept)
    for _ in 1:num_sweeps
        tr, a = mh(tr, block)
        acc += a
    end
    return tr, acc
end
# Untimed validation program: the same chain, also recording every sweep's decision.
function kernel_record(model, args, obs, num_sweeps)
    tr, _ = generate(model, args, obs)
    dec = falses(num_sweeps)
    block = select(:slope, :intercept)
    for k in 1:num_sweeps
        tr, a = mh(tr, block)
        dec[k] = a
    end
    return tr, dec
end

lp(x, mu, s) = -0.5 * log(2π) - log(s) - (x - mu)^2 / (2s^2)
slope0, intercept0 = -0.9, 1.8
hand =
    lp(slope0, 0.0, 1.0) +
    lp(intercept0, 0.0, 2.0) +
    sum(lp.(ys, slope0 .* xs .+ intercept0, 0.1))

@printf(
    "config: row %s N=%d sweeps=%d reps=%d encodings=%s\n",
    ROW,
    N,
    NUM_SWEEPS,
    NREPS,
    join(ENCODINGS, ",")
)
for enc in ENCODINGS
    model, args, obs = setup(enc)
    label = "GENJL linreg_tutorial_$(ROW) $(enc)"
    Random.seed!(1)
    kernel(model, args, obs, NUM_SWEEPS)   # warm-up, discarded
    walls = Float64[]
    local tr, acc
    for _ in 1:NREPS
        t = @elapsed begin
            tr, acc = kernel(model, args, obs, NUM_SWEEPS)
        end
        push!(walls, t)
        @printf("%s REP wall %.6f s | per-move %.4f us\n", label, t, t / NUM_SWEEPS * 1e6)
    end
    @printf(
        "%s per-move median %.4f us min %.4f us | accepts %d/%d | slope=%.10f intercept=%.10f\n",
        label,
        median(walls) / NUM_SWEEPS * 1e6,
        minimum(walls) / NUM_SWEEPS * 1e6,
        acc,
        NUM_SWEEPS,
        tr[:slope],
        tr[:intercept]
    )
    # joint density of a fixed trace
    cm = merge(obs, choicemap((:slope, slope0), (:intercept, intercept0)))
    trf, _ = generate(model, args, cm)
    sc = get_score(trf)
    @printf(
        "%s joint density: gen=%.12f hand=%.12f |diff|=%.3e\n",
        label,
        sc,
        hand,
        abs(sc - hand)
    )
    Random.seed!(1)
    trr, dec = kernel_record(model, args, obs, NUM_SWEEPS)
    println(
        "$label record: accepts $(count(dec))/$(NUM_SWEEPS), accepted sweeps $(findall(dec)), ",
        "final slope=$(repr(trr[:slope])) intercept=$(repr(trr[:intercept]))",
    )
end

# Prior invariance of the block move (no data): the same kernel on a prior-only
# model. Gen's mh(tr, selection) removes the proposal density, so every proposal
# is accepted and the prior is invariant: slope mean 0 var 1, intercept mean 0
# var 4 (an acceptance counting the prior twice would give variances 0.5 and 2).
const PI_SWEEPS = parse(Int, get(ENV, "GEN_LT_PI_SWEEPS", "200000"))
@gen function line_prior()
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    return slope
end
if PI_SWEEPS > 0
    Random.seed!(2)
    trp = simulate(line_prior, ())
    m1 = zeros(2)
    m2 = zeros(2)
    nacc = 0
    blk = select(:slope, :intercept)
    for _ in 1:PI_SWEEPS
        global trp, nacc
        trp, a = mh(trp, blk)
        nacc += a
        v = [trp[:slope], trp[:intercept]]
        m1 .+= v
        m2 .+= v .^ 2
    end
    mu = m1 ./ PI_SWEEPS
    va = m2 ./ PI_SWEEPS .- mu .^ 2
    @printf(
        "GENJL linreg_tutorial_%s prior invariance: sweeps=%d accepts=%d slope mean=%.4f var=%.4f intercept mean=%.4f var=%.4f (correct 0/1, 0/4; prior^2 target 0/0.5, 0/2)\n",
        ROW,
        PI_SWEEPS,
        nacc,
        mu[1],
        va[1],
        mu[2],
        va[2]
    )
end
