using Gen, Random, Printf
const N = 20
const X = [-3.0, -2.6842, -2.3684, -2.0526, -1.7368, -1.4211, -1.1053, -0.7895, -0.4737, -0.1579, 0.1579, 0.4737, 0.7895, 1.1053, 1.4211, 1.7368, 2.0526, 2.3684, 2.6842, 3.0]
const Y = [16.5608, 15.9844, 14.4534, 13.875, 35.0582, 12.9817, 12.0972, 11.4781, 11.3252, 10.3109, 9.7693, 9.9745, -10.6448, 6.9002, 7.2688, 6.3987, 5.8039, 5.0438, 22.2934, 3.4963]
@gen function outlier_model(xs, N)
    slope ~ normal(0.0, 2.0)
    intercept ~ normal(0.0, 10.0)
    sig = Vector{Float64}(undef, N)
    for i in 1:N
        o = ({(:o, i)} ~ bernoulli(0.1))
        sig[i] = o ? 10.0 : 1.0
    end
    ys ~ broadcasted_normal(slope .* xs .+ intercept, sig)
end
obs = choicemap((:ys, Y), (:slope, -2.0), (:intercept, 10.0))
const NUM_SWEEPS = 200
const MOVES = NUM_SWEEPS * (N + 2)
function kern()
    tr, _ = generate(outlier_model, (X, N), obs)
    for _ in 1:NUM_SWEEPS
        tr, _ = mh(tr, select(:slope))
        tr, _ = mh(tr, select(:intercept))
        for i in 1:N
            tr, _ = mh(tr, select((:o, i)))
        end
    end
    return tr
end
Random.seed!(777)
tr = kern()
best = Inf
for _ in 1:3
    t = @elapsed begin
        global tr = kern()
    end
    global best = min(best, t)
end
flags = [Int(tr[(:o, i)]) for i in 1:N]
@printf("GENJL outlier per-move %.3f us | slope = %.3f | flags = %s\n",
    best / MOVES * 1e6, tr[:slope], join(findall(==(1), flags), ","))
