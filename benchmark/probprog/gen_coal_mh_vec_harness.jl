using Gen, Random, Printf
using Gen: loggamma
const T = 56
const Y = Int[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]

struct VecPoisson <: Gen.Distribution{Vector{Int}} end
const vec_poisson = VecPoisson()
Gen.random(::VecPoisson, lam::Vector{Float64}) = [Gen.random(Gen.poisson, l) for l in lam]
function Gen.logpdf(::VecPoisson, y::Vector{Int}, lam::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(y)
        s += y[i] * log(lam[i]) - lam[i] - loggamma(y[i] + 1.0)
    end
    return s
end
Gen.logpdf_grad(::VecPoisson, y, lam) = (nothing, nothing)
Gen.has_output_grad(::VecPoisson) = false
Gen.has_argument_grads(::VecPoisson) = (false,)
Gen.is_discrete(::VecPoisson) = true
(::VecPoisson)(lam) = Gen.random(vec_poisson, lam)

@gen function coal()
    s ~ uniform_discrete(0, T)
    l1 ~ gamma(3.0, 1.0)
    l2 ~ gamma(3.0, 1.0)
    lam = Float64[t < s ? l1 : l2 for t in 0:(T - 1)]
    ys ~ vec_poisson(lam)
end
obs = choicemap()
obs[:ys] = Y
obs[:l1] = 3.0; obs[:l2] = 0.9
const NUM_SWEEPS = 2000
function kern()
    tr, _ = generate(coal, (), obs)
    hist = zeros(Int, T + 1)
    for _ in 1:NUM_SWEEPS
        tr, _ = mh(tr, select(:s))
        tr, _ = mh(tr, select(:l1))
        tr, _ = mh(tr, select(:l2))
        hist[tr[:s] + 1] += 1
    end
    return hist
end
Random.seed!(12345)
hist = kern()
best = Inf
for _ in 1:3
    t = @elapsed begin
        global hist = kern()
    end
    global best = min(best, t)
end
smap = argmax(hist) - 1
@printf("GENJL-VEC coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))
