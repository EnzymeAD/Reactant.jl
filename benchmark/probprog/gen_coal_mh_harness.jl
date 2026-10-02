using Gen, Random, Printf
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
@gen function coal()
    s ~ uniform_discrete(0, T)
    l1 ~ gamma(3.0, 1.0)
    l2 ~ gamma(3.0, 1.0)
    for t in TIDX
        {(:y, t)} ~ poisson(t < s ? l1 : l2)
    end
end
obs = choicemap()
for (i, t) in enumerate(TIDX); obs[(:y, t)] = Y[i]; end
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
for _ in 1:3
    t = @elapsed begin
        global hist = kern()
    end
    global best = min(best, t)
end
smap = argmax(hist) - 1
@printf("GENJL coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))
