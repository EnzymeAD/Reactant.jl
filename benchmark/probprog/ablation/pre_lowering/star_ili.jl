#!/usr/bin/env julia
#=
STAR: Space-Time Autoregression on US State ILINet Data
========================================================

Provenance-bearing companion row for the CONSTRUCTION mechanism of
tab:prelowering (the role Projected Dynamics played, with a real weight matrix
and a real series instead of `randn` inputs).

Model (STAR(1), Pfeifer & Deutsch 1980; Cliff-Ord lineage):

    y_{t+1} | y_t  ~  Normal( K(rho, phi) y_t ,  sigma^2 I )
    K(rho, phi)    =  rho * W  +  phi * I

W is the row-standardised contiguity matrix of the ILINet-reporting contiguous
US states, y_t is the vector of log ILI rates in week t, and the three sampled
parameters are the spatial autoregressive coefficient rho, the temporal
autoregressive coefficient phi and the log innovation scale z_sigma.

Scale-parameter note (documented choice, not an oversight): this is the
standard CONDITIONAL (one-step-ahead) STAR Gaussian likelihood, in which the
innovation scale and a separate observation scale are not identified without a
latent-state filter. Adding a Kalman filter to separate them would replace the
in-loop operation with per-step covariance recursions, i.e. a different
mechanism from the one this row exists to measure. So one scale is sampled, and
it is the innovation scale.

Mechanism: K(rho, phi) is loop-invariant but is assembled INSIDE the traced
time loop in source, so the naive program rebuilds an N x N matrix every step
and, after AD, carries a per-iteration adjoint for that construction. Pre-AD
LICM hoists the construction out; the loop keeps only the loop-variant
O(N^2) mat-vec y_t K and the residual accumulation. Because the eliminated
construction is O(N^2) rather than a factorization, the expected speedup is
modest, in the range Projected Dynamics occupied. That is the point of the row:
it is the small-mechanism companion to the factorization rows.

Data: benchmark/probprog/pplbench/pplbench/models/real_data/star_ili_states.json,
built by build_star_ili.py from (a) the US Census Bureau county adjacency file
and (b) CDC ILINet weekly state-level ILI rates via the Delphi Epidata mirror.
Both W and Y are real observed data; nothing is simulated.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/star_ili.jl <A|B> [bench|ir]
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const MODE = length(ARGS) >= 2 ? ARGS[2] : "bench"
# "recur" (default) = the state-space STAR recurrence the row measures.
# "cond"  = the conditional one-step-ahead likelihood; kept so the loop-raising
#           finding documented in the results file stays reproducible.
const FORM = get(ENV, "STAR_FORM", "recur")
const NUM_WARMUP = parse(Int, get(ENV, "PRELOWER_WARMUP", "200"))
@assert ARM in ("A", "B") "arm must be A or B (got $ARM)"
@assert FORM in ("recur", "cond") "STAR_FORM must be recur or cond (got $FORM)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]

# ── Data ─────────────────────────────────────────────────────────────────────
const DATA_PATH = joinpath(@__DIR__, "..", "..", "pplbench", "pplbench", "models",
                           "real_data", "star_ili_states.json")
const STATE_PATH = get(ENV, "PRELOWER_STATE",
                       joinpath(@__DIR__, "adapted_states", "star_$(FORM).json"))
const RAW = JSON3.read(read(DATA_PATH, String))
const N = Int(RAW["N"])
const T_STEPS = Int(RAW["T"])
const LOG2PI = log(2 * pi)

# W is row-standardised and therefore NOT symmetric, so the row-vector form of
# the spatial lag needs W'. Storing the transpose keeps the transpose out of the
# traced loop: with WT = W', `y * (rho*WT + phi*I)` equals
# `(rho*(W y) + phi*y)'` elementwise.
const W_MAT = permutedims(reduce(hcat, [collect(Float64, r) for r in RAW["W"]]))
const WT_MAT = permutedims(W_MAT)
const Y_MAT = permutedims(reduce(hcat, [collect(Float64, r) for r in RAW["Y"]]))
@assert size(W_MAT) == (N, N) "W shape $(size(W_MAT)) != ($N, $N)"
@assert size(Y_MAT) == (T_STEPS, N) "Y shape $(size(Y_MAT)) != ($T_STEPS, $N)"

# ── Model: STAR(1) conditional Gaussian log joint density ────────────────────
function star_logpdf(position, WT, Y)
    rho = @allowscalar position[1]
    phi = @allowscalar position[2]
    z_sigma = @allowscalar position[3]
    sigma = exp(z_sigma)

    I_n = Matrix{Float64}(I, N, N)
    # The loop carry must START as a traced array: a plain `zeros(1, N)` would
    # fix the carry type to Matrix{Float64} and the traced body value would then
    # be converted back on every iteration (scalar-indexing error). Deriving the
    # zero from Y keeps it traced. Per-state squared residuals are accumulated
    # and summed after the loop, which is the same SSE.
    acc = 0.0 .* Y[1:1, :]                              # 1 x N, traced zeros

    if FORM == "recur"
        # State-space reading of the STAR recurrence: the latent field is
        # propagated deterministically by K from the first observed week and
        # every week is observed with Normal(x_t, sigma) error. `x` is genuinely
        # loop-carried, so this is a recurrence, not a map over t.
        # Static 2-D slice (+ fresh buffer), not `reshape(Y[1, :], 1, N)`: the
        # latter is a lazy Base.ReshapedArray wrapper and the loop carry would
        # fail to convert the TracedRArray the body produces back to it.
        x = Y[1:1, :] .+ 0.0                            # 1 x N, real week-1 field
        @trace track_numbers = false for t in 0:(T_STEPS - 2)
            # Transition construction: LOOP-INVARIANT N x N assembly.
            K = rho .* WT .+ phi .* I_n                 # N x N, O(N^2)
            x = x * K                                    # 1 x N, loop-CARRIED
            ynext = reshape(Y[t + 2, :], 1, N)
            resid = ynext .- x
            acc = acc .+ resid .* resid
        end
    else
        # Conditional (one-step-ahead) STAR likelihood. Kept for the record: it
        # has NO loop-carried state, so the loop is a map-reduce over t and
        # Reactant's loop-raising rewrites the whole thing into a single batched
        # (T-1) x N by N x N matmul in BOTH arms. There is then nothing in a
        # loop for LICM to hoist and the two arms compile to identical IR. See
        # the results file; this is a property of the formulation, not a gap in
        # the LICM pattern set.
        @trace track_numbers = false for t in 0:(T_STEPS - 2)
            K = rho .* WT .+ phi .* I_n
            yt = reshape(Y[t + 1, :], 1, N)
            ynext = reshape(Y[t + 2, :], 1, N)
            resid = ynext .- yt * K
            acc = acc .+ resid .* resid
        end
    end

    sse = sum(acc)
    n_obs = (T_STEPS - 1) * N
    loglik = -1.0 * n_obs * (0.5 * LOG2PI + z_sigma) - 0.5 * sse / (sigma * sigma)
    logprior = -0.5 * (rho * rho + phi * phi + z_sigma * z_sigma) - 1.5 * LOG2PI
    return logprior + loglik
end

function run_mcmc(rng, pos, ss, imm, WT, Y; num_samples=20)
    ProbProg.mcmc_logpdf(rng, star_logpdf, pos, WT, Y;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# Warmup phase of the frozen-adapted protocol: adaptation ON, not timed.
function run_warmup(rng, pos, ss, imm, WT, Y)
    ProbProg.mcmc_logpdf(rng, star_logpdf, pos, WT, Y;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=NUM_WARMUP, num_samples=1,
        adapt_step_size=true, adapt_mass_matrix=true)
end

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = ARM_DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = ARM_DISABLE_SICM
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=ARM_DISABLE_LICM,
    sync=true,
)

println("=== STAR (ILINet): arm $ARM (path ",
        ARM_DISABLE_OPT ? "flat" : "region",
        ", licm ", ARM_DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", ARM_DISABLE_SICM ? "OFF" : "ON",
        "), form $FORM, N=$N T=$T_STEPS, mode=$MODE ===")
println("  data: $(normpath(DATA_PATH))")
println("  states=$(length(RAW["states"])) edges=$(RAW["n_edges"]) ",
        "epiweeks $(RAW["epiweeks"][1])..$(RAW["epiweeks"][end])")

WT_r = Reactant.to_rarray(WT_MAT)
Y_r = Reactant.to_rarray(Y_MAT)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 3, 3))
pos0 = Reactant.to_rarray(reshape([0.0, 0.0, 0.0], 1, 3))
const STEP_SIZE = parse(Float64, get(ENV, "STAR_STEP", "0.02"))
ss = Reactant.ConcreteRNumber(STEP_SIZE)

# Frozen-adapted protocol: `warmup` writes the state, `bench` consumes it. Both
# arms read the SAME file, so they share step size, inverse mass and position.
if MODE == "bench"
    ST = load_adapted_state(STATE_PATH)
    ss = Reactant.ConcreteRNumber(ST.step)
    imm_r = Reactant.to_rarray(ST.imm)
    pos0 = Reactant.to_rarray(ST.pos)
    println("  frozen-adapted state: $(STATE_PATH)")
    println("    step=$(ST.step)  inv_mass_rank=$(ndims(ST.imm))  ",
            "warmup=$(ST.meta["num_warmup"]) by arm $(ST.meta["produced_by_arm"])")
end

# ── Modes ────────────────────────────────────────────────────────────────────
if MODE == "warmup"
    @assert ARM == "B" "warmup must run in the Pre-Opt arm (B); got $ARM"
    seed = UInt64[42, 0]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    println("  warmup: $NUM_WARMUP iterations, adaptation ON (step size + mass matrix)")
    ct = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_warmup(
            rng, pos0, ss, imm_r, WT_r, Y_r)
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed ((_, _, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, WT_r, Y_r))
    println("  Warmup run: $(round(wt, digits=1))s")
    save_adapted_state(STATE_PATH, "star_$(FORM)", extract_adapted_state(st);
        num_warmup=NUM_WARMUP, produced_by_arm=ARM, warmup_seed=seed,
        initial_step_size=STEP_SIZE,
        extra=Dict{String,Any}("N" => N, "T" => T_STEPS, "form" => FORM,
                               "max_tree_depth" => 5))

elseif MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos0, ss, imm_r, WT_r, Y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "star_$(FORM)_arm$(ARM).mlir"), hlo_str)
else  # bench
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos0, ss, imm_r, WT_r, Y_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos0, ss, imm_r, WT_r, Y_r)

    n_runs = parse(Int, get(ENV, "PRELOWER_NRUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos0, ss, imm_r, WT_r, Y_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    geo_t = exp(sum(log.(times)) / length(times))
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  Geomean (20 NUTS samples): $(round(geo_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    VSEED = UInt64[parse(UInt64, get(ENV, "PRELOWER_VERIFY_SEED", "7")), 11]
    rng_v = ProbProg.ReactantRNG(Reactant.to_rarray(copy(VSEED)))
    samples, diagnostics, log_densities, _, _ = compiled_fn(rng_v, pos0, ss, imm_r, WT_r, Y_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println("  verification (seed $(Int.(VSEED))): n_accepted=$(sum(diag_arr[:, 1]))",
            " n_diverging=$(sum(diag_arr[:, 2]))")
    println("  last sample (rho, phi, z_sigma): $(samples_arr[end, :])")

    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos0, ss, imm_r, WT_r, Y_r)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "star_$(FORM)_arm$(ARM).mlir"), hlo_str)

    write_cell(joinpath(@__DIR__, "logs", "cells", "star_$(FORM)_arm$(ARM).json"),
        Dict(
            "model" => "star_ili", "form" => FORM,
            "size" => "N$(N)_T$(T_STEPS)", "arm" => ARM,
            "N" => N, "T" => T_STEPS, "n_edges" => RAW["n_edges"],
            "epiweek_first" => RAW["epiweeks"][1], "epiweek_last" => RAW["epiweeks"][end],
            "step_size" => STEP_SIZE, "max_tree_depth" => 5, "num_samples" => 20,
            "compile_s" => compile_time,
            "times_ms" => times .* 1000, "median_ms" => median_t * 1000,
            "verify_seed" => Int.(VSEED),
            "samples" => [collect(samples_arr[j, :]) for j in 1:size(samples_arr, 1)],
            "accepted" => Int.(diag_arr[:, 1]),
            "diverging" => Int.(diag_arr[:, 2]),
            "log_densities" => collect(ld_arr),
            "ir" => counts,
        ))
end
