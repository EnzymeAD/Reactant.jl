# Coal-mining change point, masked-rate encoding with the FULL Poisson log density
# (2026-10-03, figure 13 round 2, oopsla/evidence/fig13_round2_prep_2026-10-03).
# Same model, data, start, seed and kernel as impulse_coal_mh_masked.jl, with two
# changes so every system computes the same per-move terms on this row:
#   * the observation's log density computes -lgamma(y + 1) of every count in the
#     model, as Gen.jl's VecPoisson and GenJAX's tfp Poisson do in every move
#     (impulse_coal_mh_masked.jl passed the log-factorials in, precomputed on the
#     host);
#   * the switch histogram adds one to a single bin (dynamic update at s), as
#     GenJAX's hist.at[s].add(1) and Gen.jl's hist[s + 1] += 1 do, instead of a
#     compare-select-add over all T + 1 bins.
using Reactant, Random, LinearAlgebra, Statistics, Printf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, TracedRNumber
# Stock Base.wait (blocks, like JAX's block_until_ready): a spin-yield override competed with the PJRT worker thread on one pinned CPU.
Reactant.set_default_backend("cpu")
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
    const T, _tidx, _counts = load_coal_jarrett()
    y_data = Float64.(_counts)
    t_data = Float64.(_tidx)
else
    COAL_DATA == "simulated" || error("COAL_DATA must be simulated or jarrett")
    const T = 56
    y_data = Float64[6, 3, 1, 7, 3, 3, 5, 5, 3, 4, 5, 2, 3, 1, 3, 2, 3, 4, 6, 2, 1, 2, 7, 1, 4, 0, 5, 2, 2, 2, 2, 7, 3, 7, 2, 3, 4, 5, 4, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 0]
    t_data = Float64.(0:(T - 1))
end

# Masked (rate-switch) encoding: one vector Poisson site, GenJAX-equivalent.
_y_sampler(rng, s, l1, l2, tvec) = zero.(tvec)
function _y_logpdf(yv, s, l1, l2, tvec)
    lam = ifelse.(tvec .< sum(s), sum(l1), sum(l2))
    return sum(yv .* log.(lam) .- lam .- Reactant.Ops.lgamma(yv .+ 1.0))
end
function coal_masked(rng, tvec)
    _, s = ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, Float64(T)); symbol=:s)
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    _, _ = ProbProg.sample(rng, _y_sampler, s, l1, l2, tvec; symbol=:y, logpdf=_y_logpdf)
    return nothing
end
# Add one to the bin of the current switch (bins are 0..T, columns 1..T+1).
function add_switch(hist, trace)
    k = Reactant.Ops.convert(TracedRNumber{Int64}, sum(trace[1:1, 1:1])) + 1
    return Reactant.Ops.dynamic_update_slice(
        hist, Reactant.Ops.dynamic_slice(hist, [1, k], [1, 1]) .+ 1.0, [1, k])
end
function mh_prog(rng, model, tvec, bins, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(rng, obs_tensor, model, tvec; constrained_addresses=constrained)
    hist = zero.(bins)
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
            selection=ProbProg.select(ProbProg.Address(:s)))
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
            selection=ProbProg.select(ProbProg.Address(:l1)))
        trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
            selection=ProbProg.select(ProbProg.Address(:l2)))
        hist = add_switch(hist, trace)
    end
    return hist
end
tvec = Reactant.to_rarray(t_data)
bins = Reactant.to_rarray(reshape(Float64.(0:T), 1, T + 1))
NUM_SWEEPS = 2000
rng = ReactantRNG(Reactant.to_rarray(UInt64[12345, 0]))
if COAL_DATA == "jarrett"
    # Start from the prior. Impulse's generate lowering cannot draw an
    # unconstrained Gamma site (its sampler is a traced rejection loop that
    # only the simulate and regenerate lowerings call), so the program draws
    # the two starting rates itself with that same sampler and splices them
    # into the constraint; generate then draws s from its prior. Every timed
    # repetition therefore starts from a fresh prior draw, as Gen's generate
    # and GenJAX's importance do. The slots of l1 and l2 in the flattened
    # constraint are located with sentinel values.
    obs = ProbProg.Constraint(:y => y_data, :l1 => [-1.0], :l2 => [-2.0])
    constrained = ProbProg.extract_addresses(obs)
    ot = ProbProg.flatten_constraint(obs)
    _flat = vec(Array(ot))
    _i1 = findfirst(==(-1.0), _flat); _i2 = findfirst(==(-2.0), _flat)
    _onehot(i) = (v = zeros(1, length(_flat)); v[1, i] = 1.0; v)
    keep = Reactant.to_rarray(1.0 .- _onehot(_i1) .- _onehot(_i2))
    e1 = Reactant.to_rarray(_onehot(_i1)); e2 = Reactant.to_rarray(_onehot(_i2))
    function prior_start(rng, ot, keep, e1, e2)
        g = ProbProg._gamma_sampler(rng, 3.0, 1.0, (2,))
        return ot .* keep .+ e1 .* sum(g[1:1]) .+ e2 .* sum(g[2:2])
    end
    function mh_prog_prior(rng, model, tvec, bins, ot, keep, e1, e2, constrained, num_iters)
        return mh_prog(rng, model, tvec, bins, prior_start(rng, ot, keep, e1, e2),
            constrained, num_iters)
    end
    prog = mh_prog_prior
    args = (rng, coal_masked, tvec, bins, ot, keep, e1, e2, constrained, ConcreteRNumber(NUM_SWEEPS))
else
    obs = ProbProg.Constraint(:y => y_data, :l1 => [3.0], :l2 => [0.9])
    constrained = ProbProg.extract_addresses(obs)
    ot = ProbProg.flatten_constraint(obs)
    prog = mh_prog
    args = (rng, coal_masked, tvec, bins, ot, constrained, ConcreteRNumber(NUM_SWEEPS))
end
ct = @elapsed begin
    global compiled, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog prog(args...)
    end
end
# Timed span (FIX_LEDGER H1): one call of the program compiled with sync = true,
# which returns after the stock Base.wait on every output, as GenJAX's span ends
# with block_until_ready. The arguments (the sweep count included) are device
# values built once, and the host conversion of the outputs runs after the span.
@printf("compile %.1fs sync=%s\n", ct, compiled.compiled_with_sync)
hist = Array(compiled(args...))[:]   # warm-up, discarded
best = Inf
for _ in 1:20
    t = @elapsed begin
        global out = compiled(args...)
    end
    global hist = Array(out)[:]   # host conversion, untimed
    global best = min(best, t)
    @printf("IMPULSE-LGAMMA coal REP wall %.6f s | per-move %.4f us\n", t, t / (3 * NUM_SWEEPS) * 1e6)
end
smap = argmax(hist) - 1
@printf("IMPULSE-LGAMMA coal per-move %.3f us | MAP s = %d | mass[34,46] = %.4f\n",
    best / (3 * NUM_SWEEPS) * 1e6, smap, sum(hist[35:47]) / sum(hist))

if COAL_DATA == "jarrett"
    # Validation (untimed). The timed program above is unchanged; a second
    # program runs a longer chain from its own prior start and also
    # accumulates the two rates. Trace columns follow the walk order s, l1,
    # l2, y.
    function mh_prog_stats(rng, model, tvec, bins, ot, keep, e1, e2, constrained, num_iters)
        trace, weight, _ = ProbProg.generate(rng, prior_start(rng, ot, keep, e1, e2), model, tvec;
            constrained_addresses=constrained)
        hist = zero.(bins)
        l1s = zero.(bins[1:1, 1:1]); l2s = zero.(bins[1:1, 1:1])
        @trace for _ in 1:num_iters
            trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
                selection=ProbProg.select(ProbProg.Address(:s)))
            trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
                selection=ProbProg.select(ProbProg.Address(:l1)))
            trace, weight, _ = ProbProg.mh(rng, trace, weight, model, tvec;
                selection=ProbProg.select(ProbProg.Address(:l2)))
            hist = add_switch(hist, trace)
            l1s = l1s .+ trace[1:1, 2:2]
            l2s = l2s .+ trace[1:1, 3:3]
        end
        return hist, l1s, l2s
    end
    NUM_VAL = parse(Int, get(ENV, "COAL_VAL_SWEEPS", "50000"))
    @printf("COAL-HIST impulse-lgamma timed %s\n", join(Int.(round.(hist)), " "))
    vargs = (rng, coal_masked, tvec, bins, ot, keep, e1, e2, constrained, ConcreteRNumber(NUM_VAL))
    cstats, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog mh_prog_stats(vargs...)
    end
    vh, v1, v2 = cstats(vargs...)
    vh = Array(vh)[:]
    @printf("COAL-HIST impulse-lgamma val %s\n", join(Int.(round.(vh)), " "))
    @printf("COAL-VAL impulse-lgamma sweeps=%d E[l1]=%.4f E[l2]=%.4f MAP year=%d\n",
        NUM_VAL, sum(Array(v1)) / NUM_VAL, sum(Array(v2)) / NUM_VAL, 1851 + argmax(vh) - 1)
end
