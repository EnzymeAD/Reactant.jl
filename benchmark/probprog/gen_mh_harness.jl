# Gen.jl baseline for the programmable-MH benchmark (WP4 / eval:programmable).
#
# Runs the same two models and the same user-composed inference procedure as
# test/probprog/mh_sicm_benchmark.jl, in Gen.jl:
#   (A) the rebuttal snippet: plain linear regression, alternating
#       mh(select(:slope)) / mh(select(:intercept));
#   (B) the hoistable-covariance variant: slope ~ Normal(0,2), tau ~ HalfNormal(1),
#       y ~ MVN(slope*xs, tau^2*R), alternating mh(select(:slope)) / mh(select(:tau)).
# Gen's mh(trace, selection) is the same algorithm as ProbProg.mh with a selection
# (prior-regenerate proposal, MH accept on the weight ratio), so per-move
# wall-clock is directly comparable. Data, densities, and move counts match the
# Impulse benchmark exactly; both the dynamic and the static Gen modeling
# language are measured.
#
# Protocol (mirrors run_arm): one warm run of the full kernel (generate + loop),
# then per-move = min wall-clock over 8 reps / (2 * num_iters).
#
# Run:
#   julia --project=benchmark/probprog/gen_env benchmark/probprog/gen_mh_harness.jl
using Gen, Random, LinearAlgebra, Printf

const OUT = get(ENV, "GEN_MH_OUT", "/mnt/sbrantq/probprog/oopsla/evidence/programmable_mh")
mkpath(OUT)

# HalfNormal matching the Impulse HALF_NORMAL native (DistBuilder.cpp):
#   sample = abs(sigma * z), z ~ N(0,1)
#   log p  = log(2) - log(sigma) - 0.5*log(2*pi) - x^2/(2*sigma^2)
struct HalfNormal <: Gen.Distribution{Float64} end
const half_normal = HalfNormal()
Gen.random(::HalfNormal, sigma::Real) = abs(sigma * randn())
Gen.logpdf(::HalfNormal, x::Real, sigma::Real) =
    x < 0 ? -Inf : log(2.0) - log(sigma) - 0.5 * log(2π) - x^2 / (2 * sigma^2)
Gen.logpdf_grad(::HalfNormal, x, sigma) = (nothing, nothing)
Gen.has_output_grad(::HalfNormal) = false
Gen.has_argument_grads(::HalfNormal) = (false,)
Gen.is_discrete(::HalfNormal) = false
(::HalfNormal)(sigma) = Gen.random(half_normal, sigma)

# ---------------------------------------------------------------------------
# (A) rebuttal snippet: plain linear regression
# ---------------------------------------------------------------------------
@gen function linreg_dynamic(xs, ones_n)
    slope ~ normal(0.0, 2.0)
    intercept ~ normal(0.0, 10.0)
    ys ~ broadcasted_normal(slope .* xs .+ intercept, ones_n)
    return ys
end

linreg_mu(slope, intercept, xs) = slope .* xs .+ intercept

@gen (static) function linreg_static(xs, ones_n)
    slope ~ normal(0.0, 2.0)
    intercept ~ normal(0.0, 10.0)
    mu = linreg_mu(slope, intercept, xs)
    ys ~ broadcasted_normal(mu, ones_n)
    return ys
end

# ---------------------------------------------------------------------------
# (B) hoistable-covariance variant
# ---------------------------------------------------------------------------
@gen function cov_dynamic(xs, R)
    slope ~ normal(0.0, 2.0)
    tau ~ half_normal(1.0)
    y ~ mvnormal(slope .* xs, (tau * tau) .* R)
    return y
end

cov_mu(slope, xs) = slope .* xs
cov_K(tau, R) = (tau * tau) .* R

@gen (static) function cov_static(xs, R)
    slope ~ normal(0.0, 2.0)
    tau ~ half_normal(1.0)
    mu = cov_mu(slope, xs)
    K = cov_K(tau, R)
    y ~ mvnormal(mu, K)
    return y
end

# ---------------------------------------------------------------------------
# Data (identical to mh_sicm_benchmark.jl)
# ---------------------------------------------------------------------------
function data_linreg()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
    return (xs=xs, ys=ys)
end
function data_cov(n)
    xs = collect(range(-1.0, 1.0; length=n))
    R = [i == j ? 1.0 : 0.3^abs(i - j) for i in 1:n, j in 1:n]
    y = -2.0 .* xs .+ 0.15 .* collect(range(0.4, -0.4; length=n))
    return (xs=xs, R=R, y=y)
end

# ---------------------------------------------------------------------------
# Kernels: the rebuttal's user-composed alternating single-address MH loop
# ---------------------------------------------------------------------------
function kernel(model, args, obs, addr1, addr2, num_iters)
    tr, _ = generate(model, args, obs)
    acc = 0
    s1 = select(addr1)
    s2 = select(addr2)
    for _ in 1:num_iters
        tr, a1 = mh(tr, s1); acc += a1
        tr, a2 = mh(tr, s2); acc += a2
    end
    return tr, acc
end

function time_arm(f; reps::Int=8)
    out = f()   # warm
    best = Inf
    for _ in 1:reps
        t = @elapsed f()
        best = min(best, t)
    end
    return best, out
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
const NUM_ITERS = 500
const NUM_STEPS = 2 * NUM_ITERS

dA = data_linreg()
obsA = choicemap((:ys, dA.ys))
dB = data_cov(8)
obsB = choicemap((:y, dB.y))
ones_n = fill(1.0, length(dA.xs))

results = Dict{String,Any}()
for (label, model, args, obs, a1, a2) in (
    ("linreg_dynamic", linreg_dynamic, (dA.xs, ones_n), obsA, :slope, :intercept),
    ("linreg_static", linreg_static, (dA.xs, ones_n), obsA, :slope, :intercept),
    ("cov_dynamic", cov_dynamic, (dB.xs, dB.R), obsB, :slope, :tau),
    ("cov_static", cov_static, (dB.xs, dB.R), obsB, :slope, :tau),
)
    Random.seed!(1)
    best, (tr, acc) = time_arm(() -> kernel(model, args, obs, a1, a2, NUM_ITERS))
    per_move_us = best / NUM_STEPS * 1e6
    ch = get_choices(tr)
    v1 = ch[a1]
    v2 = ch[a2]
    results[label] = (per_move_us=per_move_us, accepts=acc, v1=v1, v2=v2)
    @printf("%-16s per-move %8.3f us | accepts %4d/%d | %s=%.6f %s=%.6f\n",
        label, per_move_us, acc, NUM_STEPS, a1, v1, a2, v2)
end

open(joinpath(OUT, "gen_baseline.json"), "w") do io
    println(io, "{")
    println(io, "  \"julia\": \"$(VERSION)\", \"gen\": \"0.4.8 (local checkout)\",")
    println(io, "  \"num_iters\": $NUM_ITERS, \"mh_steps\": $NUM_STEPS, \"reps\": 8,")
    entries = String[]
    for (k, r) in sort(collect(results); by=first)
        push!(entries, """  "$k": {"per_move_us": $(round(r.per_move_us, digits=4)), "accepts": $(r.accepts), "addr1": $(r.v1), "addr2": $(r.v2)}""")
    end
    println(io, join(entries, ",\n"))
    println(io, "}")
end

open(joinpath(OUT, "gen_baseline.md"), "w") do io
    println(io, "| model | Gen arm | per-move (µs) | accepts /1000 |")
    println(io, "|---|---|---:|---:|")
    for (k, r) in sort(collect(results); by=first)
        @printf(io, "| %s | %s | %.3f | %d |\n",
            startswith(k, "cov") ? "scale_linreg_mvn (n=8)" : "linreg (n=10)",
            endswith(k, "static") ? "static DSL" : "dynamic DSL",
            r.per_move_us, r.accepts)
    end
end

# ---------------------------------------------------------------------------
# n-sweep on the covariance model: Gen re-runs chol(tau^2*R) (O(n^3)) inside
# every move's density; SICM reduces Impulse's per-move cost to O(n^2). The
# sweep exposes the asymptotic separation that n=8 hides.
# ---------------------------------------------------------------------------
sweep = Dict{Int,Any}()
for n in (8, 32, 64, 128, 256)
    d = data_cov(n)
    obs = choicemap((:y, d.y))
    Random.seed!(1)
    best, (tr, acc) = time_arm(() -> kernel(cov_dynamic, (d.xs, d.R), obs, :slope, :tau, NUM_ITERS))
    per_move_us = best / NUM_STEPS * 1e6
    sweep[n] = (per_move_us=per_move_us, accepts=acc)
    @printf("cov_dynamic n=%-4d per-move %10.3f us | accepts %4d/%d\n", n, per_move_us, acc, NUM_STEPS)
end

open(joinpath(OUT, "gen_baseline_sweep.json"), "w") do io
    println(io, "{")
    entries = String[]
    for n in sort(collect(keys(sweep)))
        r = sweep[n]
        push!(entries, """  "$n": {"per_move_us": $(round(r.per_move_us, digits=4)), "accepts": $(r.accepts)}""")
    end
    println(io, join(entries, ",\n"))
    println(io, "}")
end

println("Artifacts in ", OUT)
