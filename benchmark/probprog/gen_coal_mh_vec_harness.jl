using Gen, Random, Printf
using Gen: loggamma
# COAL_DATA selects the series:
#   simulated (default)  T = 56 simulated counts, chain warm-started at the
#                        simulation's true rates (l1 = 3.0, l2 = 0.9)
#   jarrett              the recorded British series 1851..1961 (T = 111, the
#                        two missing years dropped, every other year at its
#                        calendar index; see coal_jarrett_data.jl), chain
#                        started from the prior (only the counts constrained)
const COAL_DATA = get(ENV, "COAL_DATA", "simulated")
if COAL_DATA == "jarrett"
    include(joinpath(@__DIR__, "coal_jarrett_data.jl"))
    const T, TIDX, Y = load_coal_jarrett()
else
    COAL_DATA == "simulated" || error("COAL_DATA must be simulated or jarrett")
    const T = 56
    const Y = Int[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]
    const TIDX = 0:(T - 1)
end

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
    lam = Float64[t < s ? l1 : l2 for t in TIDX]
    ys ~ vec_poisson(lam)
end
obs = choicemap()
obs[:ys] = Y
if COAL_DATA == "simulated"
    obs[:l1] = 3.0; obs[:l2] = 0.9
end
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
for _ in 1:20
    t = @elapsed begin
        global hist = kern()
    end
    global best = min(best, t)
    @printf("GENJL-VEC coal REP wall %.6f s | per-move %.4f us\n", t, t / (3 * NUM_SWEEPS) * 1e6)
end
smap = argmax(hist) - 1
@printf("GENJL-VEC coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))

if COAL_DATA == "jarrett"
    # Validation (untimed): the timed histogram, then one longer chain that
    # also accumulates the two rates.
    @printf("COAL-HIST genjl-vec timed %s\n", join(hist, " "))
    NUM_VAL = parse(Int, get(ENV, "COAL_VAL_SWEEPS", "50000"))
    function kern_stats(n)
        tr, _ = generate(coal, (), obs)
        h = zeros(Int, T + 1); l1s = 0.0; l2s = 0.0
        for _ in 1:n
            tr, _ = mh(tr, select(:s))
            tr, _ = mh(tr, select(:l1))
            tr, _ = mh(tr, select(:l2))
            h[tr[:s] + 1] += 1; l1s += tr[:l1]; l2s += tr[:l2]
        end
        return h, l1s / n, l2s / n
    end
    vh, m1, m2 = kern_stats(NUM_VAL)
    @printf("COAL-HIST genjl-vec val %s\n", join(vh, " "))
    @printf("COAL-VAL genjl-vec sweeps=%d E[l1]=%.4f E[l2]=%.4f MAP year=%d\n",
        NUM_VAL, m1, m2, 1851 + argmax(vh) - 1)
end
