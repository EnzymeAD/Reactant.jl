using Reactant, Random, LinearAlgebra, Statistics, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
# Stock Base.wait (blocks, like JAX's block_until_ready): a spin-yield override competed with the PJRT worker thread on one pinned CPU.
Reactant.set_default_backend("cpu")
const N = 20
const X = [-3.0, -2.6842, -2.3684, -2.0526, -1.7368, -1.4211, -1.1053, -0.7895, -0.4737, -0.1579, 0.1579, 0.4737, 0.7895, 1.1053, 1.4211, 1.7368, 2.0526, 2.3684, 2.6842, 3.0]
const Y = [16.5608, 15.9844, 14.4534, 13.875, 35.0582, 12.9817, 12.0972, 11.4781, 11.3252, 10.3109, 9.7693, 9.9745, -10.6448, 6.9002, 7.2688, 6.3987, 5.8039, 5.0438, 22.2934, 3.4963]

_y_sampler(rng, mu, sig, lg) = zero.(mu)
_y_logpdf(yv, mu, sig, lg) = sum(.-log.(sig) .- 0.5 * log(2π) .- (yv .- mu) .^ 2 ./ (2 .* sig .^ 2))

const LOGIT01 = log(0.1 / 0.9)
const EBAS = [Float64.((1:N) .== i) for i in 1:N]
function outlier_model(rng, xs, lg)
    _, slope = ProbProg.sample(rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:slope)
    _, intercept = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:intercept)
    oacc = zero.(xs)
    for i in 1:N
        _, o = ProbProg.sample(rng, ProbProg.Bernoulli(LOGIT01, (1,)); symbol=Symbol("o_", i))
        oacc = oacc .+ EBAS[i] .* sum(o)
    end
    sig = 1.0 .+ 9.0 .* oacc
    mu = sum(slope) .* xs .+ sum(intercept)
    _, _ = ProbProg.sample(rng, _y_sampler, mu, sig, lg; symbol=:y, logpdf=_y_logpdf)
    return nothing
end

const FLAG_MOVES = [
    :((trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs, lg;
        selection=ProbProg.select(ProbProg.Address($(QuoteNode(Symbol("o_", i)))))))
    for i in 1:N
]
@eval function sweep_prog(rng, model, xs, lg, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(rng, obs_tensor, model, xs, lg; constrained_addresses=constrained)
    @trace for _ in 1:num_iters
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs, lg;
            selection=ProbProg.select(ProbProg.Address(:slope)))
        (trace, weight, _) = ProbProg.mh(rng, trace, weight, model, xs, lg;
            selection=ProbProg.select(ProbProg.Address(:intercept)))
        $(FLAG_MOVES...)
    end
    return trace, weight
end

xs = Reactant.to_rarray(X)
lg = Reactant.to_rarray(zero.(Y))
obs = ProbProg.Constraint(:y => Y, :slope => [-2.0], :intercept => [10.0])
constrained = ProbProg.extract_addresses(obs)
ot = ProbProg.flatten_constraint(obs)
NUM_SWEEPS = 200
MOVES = NUM_SWEEPS * (N + 2)
rng = ReactantRNG(Reactant.to_rarray(UInt64[777, 0]))
ct = @elapsed begin
    global compiled, tt = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog sweep_prog(
            rng, outlier_model, xs, lg, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
    end
end
# Timed span (2026-10-03, FIX_LEDGER H1): one call of the program compiled with
# sync = true, which returns after the stock Base.wait on every output, as
# GenJAX's span ends with block_until_ready. The sweep count is converted to a
# device scalar once, and the host conversion of the outputs runs after the span.
@printf("compile %.1fs sync=%s\n", ct, compiled.compiled_with_sync)
iters = ConcreteRNumber(NUM_SWEEPS)
out = compiled(rng, outlier_model, xs, lg, ot, constrained, iters)
Array(out[1])   # warm-up, discarded
best = Inf
for _ in 1:20
    t = @elapsed begin
        global out = compiled(rng, outlier_model, xs, lg, ot, constrained, iters)
    end
    Array(out[1])   # host conversion, untimed
    global best = min(best, t)
    @printf("IMPULSE outlier REP wall %.6f s | per-move %.4f us\n", t, t / MOVES * 1e6)
end
trace = ProbProg.unflatten_trace(out[1], out[2], tt.entries, ())
slope_f = trace.choices[:slope][1, 1]
flags = [Int(trace.choices[Symbol("o_", i)][1, 1] > 0.5) for i in 1:N]
@printf("IMPULSE outlier per-move %.3f us | slope = %.3f | flags = %s\n",
    best / MOVES * 1e6, slope_f, join(findall(==(1), flags), ","))
