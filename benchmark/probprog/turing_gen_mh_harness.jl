using Turing, Distributions, Random, Printf, LinearAlgebra
const T = 56
const Yc = Int[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]
@model function coal(y)
    s ~ DiscreteUniform(0, T)
    l1 ~ Gamma(3.0, 1.0)
    l2 ~ Gamma(3.0, 1.0)
    lam = [t < s ? l1 : l2 for t in 0:(T - 1)]
    y ~ arraydist(Poisson.(lam))
end
const XS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
const YS = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
@model function linreg(xs, ys)
    slope ~ Normal(0.0, 2.0)
    intercept ~ Normal(0.0, 10.0)
    ys ~ MvNormal(slope .* xs .+ intercept, I)
end
function run_arm(model, gibbs, nmoves_per_sweep, name; sweeps=2000)
    chain = sample(model, gibbs, 10; progress=false)   # warm/compile
    best = Inf
    local ch
    for _ in 1:3
        Random.seed!(12345)
        t = @elapsed begin
            ch = sample(model, gibbs, sweeps; progress=false)
        end
        best = min(best, t)
    end
    return best / (sweeps * nmoves_per_sweep) * 1e6, ch
end
pm, ch = run_arm(coal(Yc), Gibbs(:s => MH(), :l1 => MH(), :l2 => MH()), 3, "coal")
sv = vec(ch[:s]); hist = [count(==(k), sv) for k in 0:T]
@printf("TURING coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    pm, argmax(hist) - 1, sum(hist[35:47]) / sum(hist))
pm2, ch2 = run_arm(linreg(XS, YS), Gibbs(:slope => MH(), :intercept => MH()), 2, "linreg")
@printf("TURING linreg per-move %.3f us | slope mean = %.3f\n", pm2, mean(vec(ch2[:slope])))
