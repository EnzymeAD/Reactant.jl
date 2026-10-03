# Impulse cell of the Gen-tutorial line-model row of the generality figure.
#
# Model: `line_model` of Gen's "Introduction to Modeling in Gen"
# (slope ~ normal(0, 1), intercept ~ normal(0, 2), y_i ~ normal(slope*x_i +
# intercept, 0.1)), with the observations as one vector site (the encoding the
# other figure-13 Impulse cells use). Kernel: the block prior-resimulation move
# of "Iterative Inference in Gen", mh(tr, select(:slope, :intercept)), once per
# sweep, 500 sweeps, after generate with the ys observed. The timed span is one
# call of the compiled program (generate + loop), 20 repetitions after one
# discarded warm-up; per-move = wall / 500. The program is compiled with
# sync = true, so the call returns only after the stock Base.wait on every
# output's PJRT future, and the span ends there, as GenJAX's span ends with
# block_until_ready. The host conversions of the outputs (_force) run after the
# span, untimed (2026-10-03, FIX_LEDGER H1; they had cost 24 to 35 us per call).
#
# Arms (IMPULSE_LT_ARMS, default "sicm nosicm"): both run the region pipeline
# (DEBUG_PROBPROG_DISABLE_OPT = false); "nosicm" sets DEBUG_PROBPROG_DISABLE_SICM.
# After the timed repetitions (untimed): joint-density check of a fixed trace
# against a hand-computed reference, the acceptance count, final slope and
# intercept, and with both arms the per-sweep decision and state comparison.
# With IMPULSE_LT_IR_OUT=<dir> the final HLO of both arms is written there and
# the ops inside the sweep loop with a dimension of N are counted (by
# oopsla/evidence/linreg_tutorial_mh_2026-09-29/scripts/count_loop_nlen_ops.py).
#
# Usage: julia --project=. impulse_mh_linreg_tutorial_harness.jl <data.json> <A|B|N100|N1000>
# (N100 and N1000 are the evidence-only sizes of the crossover curve, same recipe as B.)
using Reactant, Random, LinearAlgebra, Printf, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber
_force(x) = x isa AbstractArray ? Array(x) : Float64(x)
# Stock Base.wait (blocks, like JAX's block_until_ready): a spin-yield override competed with the PJRT worker thread on one pinned CPU.
Reactant.set_default_backend("cpu")

const DATA = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "linreg_tutorial_data.json")
const ROW = length(ARGS) >= 2 ? ARGS[2] : "A"
ROW in ("A", "B", "N100", "N1000") || error("row must be A, B, N100 or N1000")
const ARMS = split(get(ENV, "IMPULSE_LT_ARMS", "sicm nosicm"))
const IR_OUT = get(ENV, "IMPULSE_LT_IR_OUT", "")
# IMPULSE_LT_OBS: "trace" (default; the figure program) or "arg" (the
# observation-as-argument evidence variant defined below).
const OBS = get(ENV, "IMPULSE_LT_OBS", "trace")
OBS in ("trace", "arg") || error("IMPULSE_LT_OBS must be trace or arg")
const NUM_SWEEPS = 500
const NREPS = 20

# Minimal JSON reader for the two flat numeric vectors of one row (the data
# file is written by make_linreg_tutorial_data.jl; no JSON package in this env).
function read_row(path, row)
    s = read(path, String)
    i = findfirst("\"$row\": {", s)
    i === nothing && error("row $row not in $path")
    seg = s[last(i):end]
    vec(key) = begin
        j = findfirst("\"$key\": [", seg)
        k = findnext(']', seg, last(j))
        parse.(Float64, split(seg[(last(j) + 1):(k - 1)], ","))
    end
    return vec("xs"), vec("ys")
end
xs_data, ys_data = read_row(DATA, ROW)
const N = length(xs_data)

normal(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
function normal_logpdf(x, μ, σ, _)
    return -length(x) * log(σ) - length(x) / 2 * log(2π) -
           sum((x .- μ) .^ 2 ./ (2 .* (σ .^ 2)))
end
function line_model(rng, xs)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 1.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )
    _, ys = ProbProg.sample(
        rng,
        normal,
        slope .* xs .+ intercept,
        0.1,
        (length(xs),);
        symbol=:ys,
        logpdf=normal_logpdf,
    )
    return ys
end
function mh_line(rng, xs, num_iters, obs_tensor, constrained)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model, xs; constrained_addresses=constrained
    )
    acc = weight .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, a, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            line_model,
            xs;
            selection=ProbProg.select(
                ProbProg.Address(:slope), ProbProg.Address(:intercept)
            ),
        )
        acc = acc .+ ifelse.(a, 1.0, 0.0)
    end
    return trace, weight, acc
end
# Untimed validation program: the same chain, also recording every sweep's
# decision and state (trace columns follow the walk order slope, intercept, ys).
function mh_line_record(rng, xs, num_iters, obs_tensor, constrained, sweep_idx)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model, xs; constrained_addresses=constrained
    )
    dec = zero.(sweep_idx)
    sl = zero.(sweep_idx)
    ic = zero.(sweep_idx)
    k = sum(sweep_idx[1:1]) .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, a, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            line_model,
            xs;
            selection=ProbProg.select(
                ProbProg.Address(:slope), ProbProg.Address(:intercept)
            ),
        )
        k = k .+ 1.0
        hit = ifelse.(abs.(sweep_idx .- k) .< 0.5, 1.0, 0.0)
        dec = dec .+ hit .* ifelse.(a, 1.0, 0.0)
        sl = sl .+ hit .* sum(trace[1:1, 1:1])
        ic = ic .+ hit .* sum(trace[1:1, 2:2])
    end
    return dec, sl, ic, trace, weight
end
function joint_weight(rng, xs, obs_tensor, constrained)
    _, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model, xs; constrained_addresses=constrained
    )
    return weight
end

# Observation-as-argument variant (IMPULSE_LT_OBS=arg, evidence only): the ys
# site keeps its address and its constrained trace slot, but its density reads
# the observations from the model argument `yo`, a loop-invariant input, rather
# than from the site's value in the loop-carried trace. The joint density as a
# function of (slope, intercept) is the same. This measures the per-move ratio
# the row can reach once the observations are provably loop-invariant. (The
# `constraint` argument of ProbProg.mh is not usable for this: a constrained MH
# is kept flat by the region preflight, InlineImpulseRegions.cpp, so SICM would
# not run at all.)
_obs_sampler(rng, mu, yo) = yo .+ 0.0 .* mu
_obs_logpdf(x, mu, yo) = normal_logpdf(yo, mu, 0.1, nothing)
function line_model_argobs(rng, xs, yo)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 1.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )
    _, ys = ProbProg.sample(
        rng, _obs_sampler, slope .* xs .+ intercept, yo; symbol=:ys, logpdf=_obs_logpdf
    )
    return ys
end
function mh_line_argobs(rng, xs, yo, num_iters, obs_tensor, constrained)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model_argobs, xs, yo; constrained_addresses=constrained
    )
    acc = weight .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, a, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            line_model_argobs,
            xs,
            yo;
            selection=ProbProg.select(
                ProbProg.Address(:slope), ProbProg.Address(:intercept)
            ),
        )
        acc = acc .+ ifelse.(a, 1.0, 0.0)
    end
    return trace, weight, acc
end
# (argobs) Untimed validation program: the same chain, also recording every sweep's
# decision and state (trace columns follow the walk order slope, intercept, ys).
function mh_line_record_argobs(rng, xs, yo, num_iters, obs_tensor, constrained, sweep_idx)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model_argobs, xs, yo; constrained_addresses=constrained
    )
    dec = zero.(sweep_idx)
    sl = zero.(sweep_idx)
    ic = zero.(sweep_idx)
    k = sum(sweep_idx[1:1]) .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, a, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            line_model_argobs,
            xs,
            yo;
            selection=ProbProg.select(
                ProbProg.Address(:slope), ProbProg.Address(:intercept)
            ),
        )
        k = k .+ 1.0
        hit = ifelse.(abs.(sweep_idx .- k) .< 0.5, 1.0, 0.0)
        dec = dec .+ hit .* ifelse.(a, 1.0, 0.0)
        sl = sl .+ hit .* sum(trace[1:1, 1:1])
        ic = ic .+ hit .* sum(trace[1:1, 2:2])
    end
    return dec, sl, ic, trace, weight
end
function joint_weight_argobs(rng, xs, yo, obs_tensor, constrained)
    _, weight, _ = ProbProg.generate(
        rng, obs_tensor, line_model_argobs, xs, yo; constrained_addresses=constrained
    )
    return weight
end

xs = Reactant.to_rarray(xs_data)
obs = ProbProg.Constraint(:ys => ys_data)
constrained = ProbProg.extract_addresses(obs)
obs_tensor = ProbProg.flatten_constraint(obs)
iters = ConcreteRNumber(NUM_SWEEPS)
yo = Reactant.to_rarray(ys_data)
# Program and argument selection by OBS (the trace programs take xs only).
const KERNEL = OBS == "arg" ? mh_line_argobs : mh_line
const RECORD = OBS == "arg" ? mh_line_record_argobs : mh_line_record
const JOINT = OBS == "arg" ? joint_weight_argobs : joint_weight
margs() = OBS == "arg" ? (xs, yo) : (xs,)
seed() = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
function label(arm)
    return if OBS == "arg"
        "IMPULSE linreg_tutorial_argobs_$(ROW) $(arm)"
    else
        "IMPULSE linreg_tutorial_$(ROW) $(arm)"
    end
end
@printf(
    "config: row %s N=%d sweeps=%d reps=%d arms=%s obs=%s threads=%d\n",
    ROW,
    N,
    NUM_SWEEPS,
    NREPS,
    join(ARMS, ","),
    OBS,
    Threads.nthreads()
)

function set_arm(arm)
    arm in ("sicm", "nosicm") || error("arm must be sicm or nosicm")
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false
    return Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = (arm == "nosicm")
end

function run_arm(arm)
    set_arm(arm)
    rng = seed()
    local compiled, tt
    ct = @elapsed begin
        compiled, tt = ProbProg.with_trace() do
            @compile sync = true optimize = :probprog KERNEL(
                rng, margs()..., iters, obs_tensor, constrained
            )
        end
    end
    @printf("%s compile %.1fs sync=%s\n", label(arm), ct, compiled.compiled_with_sync)
    out = compiled(rng, margs()..., iters, obs_tensor, constrained)
    for o in out
        _force(o)
    end   # warm-up, discarded
    walls = Float64[]
    for _ in 1:NREPS
        t = @elapsed begin
            out = compiled(rng, margs()..., iters, obs_tensor, constrained)
        end
        for o in out
            _force(o)
        end   # host conversion, untimed
        push!(walls, t)
        @printf(
            "%s REP wall %.6f s | per-move %.4f us\n", label(arm), t, t / NUM_SWEEPS * 1e6
        )
    end
    tr = ProbProg.unflatten_trace(out[1], out[2], tt.entries, ())
    acc = sum(_force(out[3]))
    @printf(
        "%s per-move median %.4f us min %.4f us | accepts %d/%d | slope=%.10f intercept=%.10f\n",
        label(arm),
        median(walls) / NUM_SWEEPS * 1e6,
        minimum(walls) / NUM_SWEEPS * 1e6,
        round(Int, acc),
        NUM_SWEEPS,
        tr.choices[:slope][1, 1],
        tr.choices[:intercept][1, 1]
    )
    return (acc=acc, slope=tr.choices[:slope][1, 1], intercept=tr.choices[:intercept][1, 1])
end
results = Dict{String,Any}()
for arm in ARMS
    results[arm] = run_arm(arm)
end

# ---------------------------------------------------------------------------
# Validation (untimed)
# ---------------------------------------------------------------------------
# (a) joint density of a fixed trace: generate with every address constrained
# returns the joint log density; compare with the hand-computed value.
slope0, intercept0 = -0.9, 1.8
cobs = ProbProg.Constraint(:slope => [slope0], :intercept => [intercept0], :ys => ys_data)
ccon = ProbProg.extract_addresses(cobs)
cten = ProbProg.flatten_constraint(cobs)
lp(x, μ, σ) = -0.5 * log(2π) - log(σ) - (x - μ)^2 / (2σ^2)
hand =
    lp(slope0, 0.0, 1.0) +
    lp(intercept0, 0.0, 2.0) +
    sum(lp.(ys_data, slope0 .* xs_data .+ intercept0, 0.1))
for arm in ARMS
    set_arm(arm)
    cj, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog JOINT(seed(), margs()..., cten, ccon)
    end
    w = sum(Array(cj(seed(), margs()..., cten, ccon)))
    @printf(
        "%s joint density: impulse=%.12f hand=%.12f |diff|=%.3e\n",
        label(arm),
        w,
        hand,
        abs(w - hand)
    )
end

# (b) per-sweep decisions and states, per arm; compared between arms.
sweep_idx = Reactant.to_rarray(collect(1.0:NUM_SWEEPS))
rec = Dict{String,Any}()
for arm in ARMS
    set_arm(arm)
    cr, _ = ProbProg.with_trace() do
        @compile sync = true optimize = :probprog RECORD(
            seed(), margs()..., iters, obs_tensor, constrained, sweep_idx
        )
    end
    rout = cr(seed(), margs()..., iters, obs_tensor, constrained, sweep_idx)
    rec[arm] = (dec=Array(rout[1]), sl=Array(rout[2]), ic=Array(rout[3]))
    @printf(
        "%s record: accepts %d/%d, first accepted sweeps %s, final slope=%.10f intercept=%.10f\n",
        label(arm),
        round(Int, sum(rec[arm].dec)),
        NUM_SWEEPS,
        string(findall(>(0.5), rec[arm].dec)[1:min(10, end)]),
        rec[arm].sl[end],
        rec[arm].ic[end]
    )
end
# (No top-level globals with short names here: an @trace for body compiled
# later would pick them up as loop-carried values.)
function compare_arms(rec, results)
    ra, rb = rec["sicm"], rec["nosicm"]
    rel(x, y) = maximum(abs.(x .- y) ./ max.(abs.(y), 1e-300))
    @printf(
        "SICM-vs-noSICM: identical decisions = %s (%d differ) | max rel diff slope %.3e intercept %.3e | timed-run accepts %d vs %d\n",
        ra.dec == rb.dec,
        count(ra.dec .!= rb.dec),
        rel(ra.sl, rb.sl),
        rel(ra.ic, rb.ic),
        round(Int, results["sicm"].acc),
        round(Int, results["nosicm"].acc)
    )
end
haskey(rec, "sicm") && haskey(rec, "nosicm") && compare_arms(rec, results)

# (c) IR evidence: final HLO per arm, ops inside the sweep loop with a dim of N.
if !isempty(IR_OUT)
    mkpath(IR_OUT)
    for arm in ARMS
        set_arm(arm)
        hlo, _ = ProbProg.with_trace() do
            repr(
                Reactant.@code_hlo optimize = :probprog KERNEL(
                    seed(), margs()..., iters, obs_tensor, constrained
                )
            )
        end
        f = joinpath(IR_OUT, "row$(ROW)_obs$(OBS)_$(arm)_final_hlo.mlir")
        write(f, hlo)
        @printf("%s final HLO written to %s\n", label(arm), f)
    end
end

# (d) Prior invariance of the block move (no data): the same kernel on the
# prior-only model (slope, intercept, no ys site). A correct prior-resimulation
# MH accepts every proposal and leaves the prior invariant: slope mean 0 var 1,
# intercept mean 0 var 4. An acceptance that counts the regenerated sites'
# prior twice targets prior^2: variances 0.5 and 2.
const PI_SWEEPS = parse(Int, get(ENV, "IMPULSE_LT_PI_SWEEPS", "200000"))
function line_prior(rng)
    _, slope = ProbProg.sample(
        rng, normal, 0.0, 1.0, (1,); symbol=:slope, logpdf=normal_logpdf
    )
    _, intercept = ProbProg.sample(
        rng, normal, 0.0, 2.0, (1,); symbol=:intercept, logpdf=normal_logpdf
    )
    return nothing
end
function prior_prog(rng, pot, pcon, num_iters)
    trace, weight, _ = ProbProg.generate(rng, pot, line_prior; constrained_addresses=pcon)
    m1 = zero.(pot[1:1, 1:2])
    m2 = zero.(pot[1:1, 1:2])
    nacc = sum(pot[1:1, 1:1]) .* 0.0
    @trace for _ in 1:num_iters
        trace, weight, acc_, _ = ProbProg.mh(
            rng,
            trace,
            weight,
            line_prior;
            selection=ProbProg.select(
                ProbProg.Address(:slope), ProbProg.Address(:intercept)
            ),
        )
        v = trace[1:1, 1:2]
        m1 = m1 .+ v
        m2 = m2 .+ v .* v
        nacc = nacc .+ ifelse.(acc_, 1.0, 0.0)
    end
    return m1, m2, nacc
end
if PI_SWEEPS > 0
    pobs = ProbProg.Constraint(:slope => [0.0], :intercept => [0.0])
    pcon = ProbProg.extract_addresses(pobs)
    pot = ProbProg.flatten_constraint(pobs)
    for arm in ARMS
        set_arm(arm)
        cp, _ = ProbProg.with_trace() do
            @compile sync = true optimize = :probprog prior_prog(
                seed(), pot, pcon, ConcreteRNumber(PI_SWEEPS)
            )
        end
        pout = cp(seed(), pot, pcon, ConcreteRNumber(PI_SWEEPS))
        mu = vec(Array(pout[1])) ./ PI_SWEEPS
        va = vec(Array(pout[2])) ./ PI_SWEEPS .- mu .^ 2
        @printf(
            "%s prior invariance: sweeps=%d accepts=%d slope mean=%.4f var=%.4f intercept mean=%.4f var=%.4f (correct 0/1, 0/4; prior^2 target 0/0.5, 0/2)\n",
            label(arm),
            PI_SWEEPS,
            round(Int, sum(_force(pout[3]))),
            mu[1],
            va[1],
            mu[2],
            va[2]
        )
    end
end
