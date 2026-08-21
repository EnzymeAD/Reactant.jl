using Gen, Random, LinearAlgebra

struct HalfNormal <: Gen.Distribution{Float64} end
const half_normal = HalfNormal()
Gen.random(::HalfNormal, sigma::Real) = abs(sigma * randn())
Gen.logpdf(::HalfNormal, x::Real, sigma::Real) =
    x < 0 ? -Inf : log(2.0) - log(sigma) - 0.5 * log(2π) - x^2 / (2 * sigma^2)
Gen.logpdf_grad(::HalfNormal, x, sigma) = (nothing, nothing)
Gen.has_output_grad(::HalfNormal) = false
Gen.has_argument_grads(::HalfNormal) = (false,)
Gen.is_discrete(::HalfNormal) = false

@gen function cov_dynamic(xs, R)
    slope ~ normal(0.0, 2.0)
    tau ~ half_normal(1.0)
    y ~ mvnormal(slope .* xs, (tau * tau) .* R)
    return y
end

function data_cov(n)
    xs = collect(range(-1.0, 1.0; length=n))
    R = [i == j ? 1.0 : 0.3^abs(i - j) for i in 1:n, j in 1:n]
    y = -2.0 .* xs .+ 0.15 .* collect(range(0.4, -0.4; length=n))
    return (xs=xs, R=R, y=y)
end

d = data_cov(8)
slope0, tau0 = 0.7, 0.9

# (a) density cross-check: Gen's trace score vs hand-computed joint logpdf
cm = choicemap((:slope, slope0), (:tau, tau0), (:y, d.y))
tr, w = generate(cov_dynamic, (d.xs, d.R), cm)
gen_score = get_score(tr)

lp_slope = -0.5 * log(2π) - log(2.0) - slope0^2 / (2 * 2.0^2)
lp_tau = log(2.0) - log(1.0) - 0.5 * log(2π) - tau0^2 / 2
K = tau0^2 .* d.R
r = d.y .- slope0 .* d.xs
lp_y = -0.5 * (length(d.y) * log(2π) + logdet(K) + r' * (K \ r))
hand = lp_slope + lp_tau + lp_y
println("gen_score = $gen_score  hand = $hand  |diff| = $(abs(gen_score - hand))")

# (b) accept-count distribution over 10 seeds (500 iters = 1000 moves)
obs = choicemap((:y, d.y))
accs = Int[]
for seed in 1:10
    Random.seed!(seed)
    tr2, _ = generate(cov_dynamic, (d.xs, d.R), obs)
    acc = 0
    for _ in 1:500
        tr2, a1 = mh(tr2, select(:slope)); acc += a1
        tr2, a2 = mh(tr2, select(:tau)); acc += a2
    end
    push!(accs, acc)
end
println("accept counts /1000 over 10 seeds: ", accs)
println("min/median/max: ", minimum(accs), "/", sort(accs)[5], "/", maximum(accs))
