using Reactant, Random, LinearAlgebra, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
_force(x) = x isa AbstractArray ? Array(x) : Float64(x)
import Reactant.XLA.PJRT: Future
function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        while !isready(future); GC.safepoint(); yield(); end
    end
    return nothing
end
normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end
function model_linreg(rng, xs)
    _, slope = ProbProg.sample(rng, normal, 0.0, 2.0, (1,); symbol=:slope, logpdf=normal_logpdf)
    _, intercept = ProbProg.sample(rng, normal, 0.0, 10.0, (1,); symbol=:intercept, logpdf=normal_logpdf)
    _, ys = ProbProg.sample(rng, normal, slope .* xs .+ intercept, 1.0, (length(xs),); symbol=:ys, logpdf=normal_logpdf)
    return ys
end
function programmable_mh_linreg(rng, xs, num_iters, obs_tensor, constrained)
    trace, weight, _ = ProbProg.generate(rng, obs_tensor, model_linreg, xs; constrained_addresses=constrained)
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model_linreg, xs;
            selection=ProbProg.select(ProbProg.Address(:slope)))
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model_linreg, xs;
            selection=ProbProg.select(ProbProg.Address(:intercept)))
    end
    return trace, weight
end
xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
obs = ProbProg.Constraint(:ys => ys)
constrained = ProbProg.extract_addresses(obs)
obs_tensor = ProbProg.flatten_constraint(obs)
N = ConcreteRNumber(500)
rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
compiled, _ = ProbProg.with_trace() do
    @compile optimize = :probprog programmable_mh_linreg(rng, xs, N, obs_tensor, constrained)
end
out = compiled(rng, xs, N, obs_tensor, constrained); for o in out; _force(o); end
best = Inf
for _ in 1:8
    t = @elapsed begin
        o = compiled(rng, xs, N, obs_tensor, constrained); for x in o; _force(x); end
    end
    global best = min(best, t)
end
@printf("IMPULSE linreg per-move %.3f us\n", best / 1000 * 1e6)
