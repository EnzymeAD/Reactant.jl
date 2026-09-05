#!/usr/bin/env julia
#=
Heat-Equation Bayesian Inverse Problem (v1: custom logpdf)
==========================================================

1D heat equation inverse problem (Stuart 2010 framing). Infer log-diffusivity
z_kappa and log-noise z_sigma from noisy sensor observations of an implicit-Euler
time evolution.

    kappa = exp(z_kappa),  sigma = exp(z_sigma)
    M(kappa) = I + dt*kappa*L,   M x_{t+1} = x_t     (one cholesky + 2 solves/step)
    y_j ~ Normal(H x_{t_j}, sigma^2 I)

The cholesky of M(kappa) is loop-invariant across the T time steps but kept
INSIDE the traced loop in source. The ablation ladder peels loop levels:
  arm A (pre-lowering off, SICM off): cholesky inside hot loop, O(T*D^3)/grad
  arm B (pre-lowering on,  SICM off): cholesky hoisted to 1/grad, O(D^3+T*D^2)
  arm C (pre-lowering on,  SICM on) : EigenLift -> syevd(L), O(T*D^2)/grad
  arm D (pre-lowering off, SICM on) : SICM structurally cannot run (== arm A)

Arm mechanism (2026-08-21, rev. 2: standard-passes-only axis; supersedes both
the July unroll workaround and the earlier region-bundled arm B):
  - The tab:prelowering arms isolate the pre-lowering STANDARD passes and
    contain no SICM machinery: both arms run the flat expand path
    (`DEBUG_PROBPROG_DISABLE_OPT=true`; the region pipeline is SICM
    infrastructure and is out of scope here) and differ only in
    `disable_licm_optimization_passes` (CompileOptions; strips the `*_licm`
    pattern subset from the standard passes). With LICM off, 20 potrf sites
    sit inside the leaf time whiles (per-step O(D^3), fwd+rev); with LICM on,
    exactly 1 hoisted potrf per gradient (+1 top-level for InitHMC).
    Verified at D=100 2026-08-21. Arm R (region pipeline, SICM at zero
    iterations) is reported as production-pipeline context, not a table arm.
  - SICM must be EXPLICITLY off and VERIFIED off here (user directive
    2026-08-21): unlike the matmul ablation models, this model's
    M = I + dt*kappa*L is exactly CholeskyEigenLift's target pattern, so
    SICM would fire if enabled. Off by construction (flat path never runs
    the sicm pass; DISABLE_SICM=true besides) and checked per run: the
    ir_report's syevd count (EigenLift's signature op) must be 0 in every
    tab:prelowering cell.
  - `cholesky_licm` (LICM<stablehlo::CholeskyOp>, Enzyme-JAX EnzymeHLOOpt.cpp,
    2026-08-20) lets the loop-invariant cholesky hoist out of the real
    stablehlo.while pre-expansion; no loop unrolling is involved.
  - The post-AD `stablehlo-to-lapack` pass (2026-08-21) lowers rank-2
    cholesky/triangular_solve to LAPACK dpotrf/dtrsm custom_calls on CPU, so
    the IR evidence below counts shape-classified custom_calls, not
    stablehlo.cholesky ops.
  - Arms C/D are the SICM axis (C = region + SICM; D = flat, SICM inert).

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/heat_inverse.jl <A|B|R|C|D> <scale> [mode]
  scale: 0 (smoke D=20,T=10), 1 (D=100,T=100), 2 (D=300,T=30, SICM-tier regime),
         3 (D=300,T=100, "large" table row), 4 (D=60,T=100), 5 (D=60,T=50)
  mode : bench (default) | parity | point | ir
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "heat_inverse_data.jl"))
include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const SCALE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const MODE = length(ARGS) >= 3 ? ARGS[3] : "bench"
@assert ARM in ("A", "B", "R", "C", "D") "arm must be A/B/R/C/D (got $ARM)"
# Frozen-adapted protocol (2026-08-25), additive: `warmup` runs adaptation ON in
# the Pre-Opt arm and saves the adapted state; `bench` consumes it so BOTH arms
# are measured with adaptation OFF from the SAME step size, inverse mass and
# starting position. The other modes are untouched.
const NUM_WARMUP = parse(Int, get(ENV, "PRELOWER_WARMUP", "200"))
const STATE_PATH = get(ENV, "PRELOWER_STATE",
    joinpath(@__DIR__, "adapted_states", "heat_scale$(SCALE).json"))

# arm -> (disable_opt [flat path], disable_sicm, unroll_time_loops, disable_licm)
# The tab:prelowering axis isolates the pre-lowering STANDARD passes: both
# table arms use the flat expand path (no region pipeline -- that is SICM
# infrastructure, out of scope here) and differ only in the LICM pattern
# subset. Arm R gives the region-pipeline production number as context.
const ARM_KNOBS = Dict(
    "A" => (true, true, false, true),    # No Pre-Opt: flat expand, LICM off
    "B" => (true, true, false, false),   # Pre-Opt: flat expand, full std passes
    "R" => (false, true, false, false),  # context: region pipeline, SICM at 0
    "C" => (false, false, false, false), # region + SICM (SICM-ablation axis)
    "D" => (true, false, false, true),   # SICM structurally off (== A)
)
const ARM_DISABLE_OPT, DISABLE_SICM, ARM_UNROLL, ARM_DISABLE_LICM = ARM_KNOBS[ARM]

# ── Config / constants (baked for the traced logpdf) ─────────────────────────
const CFG = heat_config(SCALE)
const D = CFG.D
const T = CFG.T
const DT = CFG.dt
const M_OBS = CFG.M_obs
const S = CFG.S
const OBS_STEP = CFG.obs_step        # coarse steps between observations
const SENSOR_IDX = CFG.sensor_idx    # Vector{Int}, length S
const LOG2PI = log(2 * pi)

# ── Model: log joint density (returns a log-density scalar) ──────────────────
#
# Chunked form (approach b): M_obs unrolled windows, each a traced 0-based for
# loop that advances the implicit-Euler state OBS_STEP steps with the cholesky +
# two solves INSIDE the loop body. The observation likelihood is accumulated
# between windows (outside the traced loops). This mirrors cholesky_recurrence.jl
# (carry one state array, cholesky+solve in-loop) so it reverse-differentiates
# cleanly; a single-loop trajectory-store form trips Enzyme's while-cache logic.
function heat_inverse_logpdf(position, L, x0col, y_data)
    z_kappa = @allowscalar position[1]
    z_sigma = @allowscalar position[2]
    kappa = exp(z_kappa)
    sigma = exp(z_sigma)

    I_d = Matrix{Float64}(I, D, D)
    xcol = x0col                              # D x 1, state at coarse step 0

    loglik = -1.0 * (M_OBS * S) * (0.5 * LOG2PI + z_sigma)   # Normal norm const
    for j in 1:M_OBS
        # Advance OBS_STEP implicit-Euler steps to the next observation time.
        @trace track_numbers = false for t in 0:(OBS_STEP - 1)
            # M(kappa) = 1*I + (dt*kappa)*L -- the s*A + t*I form EigenLift targets.
            M = I_d .+ (DT * kappa) .* L        # D x D, SPD, loop-invariant
            F = cholesky(M)                      # O(D^3), loop-invariant
            xcol = F \ xcol                      # 2 triangular solves, D x 1
        end
        # xcol is now the state at coarse step j*OBS_STEP = obs_steps[j].
        obs_j = xcol[SENSOR_IDX, 1]              # length S (sensor readings)
        resid = y_data[j, :] .- obs_j            # length S
        loglik = loglik - 0.5 * sum(resid .* resid) / (sigma * sigma)
    end

    logprior = -0.5 * z_kappa * z_kappa - 0.5 * z_sigma * z_sigma - LOG2PI
    return logprior + loglik
end

function run_mcmc(rng, pos, ss, imm, L, x0col, y_data; num_samples=20)
    ProbProg.mcmc_logpdf(rng, heat_inverse_logpdf, pos, L, x0col, y_data;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# Warmup phase of the frozen-adapted protocol: adaptation ON, not timed.
function run_warmup(rng, pos, ss, imm, L, x0col, y_data)
    ProbProg.mcmc_logpdf(rng, heat_inverse_logpdf, pos, L, x0col, y_data;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=NUM_WARMUP, num_samples=1,
        adapt_step_size=true, adapt_mass_matrix=true)
end

# IR analysis (`count_in_out_loops` / `ir_report`) now lives in
# prelowering_common.jl, included above, so every row of tab:prelowering is
# classified by identical code. The verbatim copies that used to sit here were
# removed when heat joined the shared harness plumbing (2026-08-25).

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
# Path selection per arm (env overrides for diagnostics).
const DISABLE_OPT = get(ENV, "HEATINV_DISABLE_OPT", ARM_DISABLE_OPT ? "1" : "0") == "1"
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = DISABLE_SICM
# Arms B/C unroll the traced time loops (region path cannot reverse-AD a
# caching stablehlo.while); arms A/D keep real while loops on the flat path.
const UNROLL = parse(Int, get(ENV, "HEATINV_UNROLL", ARM_UNROLL ? "200" : "0"))
if UNROLL > 0
    Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = UNROLL
end
# Strip the LICM pattern subset from the standard passes for this compile
# (CompileOptions field; affects pre-expansion AND post-AD passes). Env
# override supports the diagnostic flat+LICM-on cell (path-neutrality check).
const DISABLE_LICM =
    get(ENV, "HEATINV_DISABLE_LICM", ARM_DISABLE_LICM ? "1" : "0") == "1"
# NOTE: the @compile macro ignores its other kwargs (incl. `sync`) when
# `compile_options` is given, so sync must be set HERE or timings measure
# only async dispatch.
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=DISABLE_LICM,
    sync=true,
)

println("=== Heat Inverse: arm $ARM (path ", DISABLE_OPT ? "flat" : "region",
        ", licm ", DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", DISABLE_SICM ? "OFF" : "ON",
        ", unroll=", UNROLL, "), scale $SCALE, ",
        "D=$D T=$T M_obs=$(CFG.M_obs) S=$(CFG.S), mode=$MODE ===")

cfg_data = heat_generate_data(CFG)
data_path = heat_data_path(SCALE)
heat_save_data(data_path, merge(CFG, cfg_data), cfg_data)
println("  data: kappa_true=$(CFG.kappa_true) sigma_true=$(CFG.sigma_true) dt=$(round(DT, sigdigits=4))")
println("  data file: $data_path")

L_r = Reactant.to_rarray(heat_laplacian(D))
x0_r = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))    # D x 1
y_r = Reactant.to_rarray(cfg_data.y)                             # M_obs x S
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 2, 2))            # dense identity mass
pos0 = Reactant.to_rarray(reshape([0.0, 0.0], 1, 2))            # z-space zeros
# Step size: 0.1 diverges every iteration on this tight posterior (still
# lockstep-matches NumPyro, but trees terminate at depth ~1); benchmarks use a
# smaller step so tree building does real work. Keep both systems identical.
const STEP_SIZE = parse(Float64, get(ENV, "HEATINV_STEP", "0.02"))
ss = Reactant.ConcreteRNumber(STEP_SIZE)

# Frozen-adapted protocol: `bench` runs from the saved adapted state so both
# arms share step size, inverse mass and starting position.
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
            rng, pos0, ss, imm_r, L_r, x0_r, y_r)
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed ((_, _, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, L_r, x0_r, y_r))
    println("  Warmup run: $(round(wt, digits=1))s")
    save_adapted_state(STATE_PATH, "heat_scale$(SCALE)", extract_adapted_state(st);
        num_warmup=NUM_WARMUP, produced_by_arm=ARM, warmup_seed=seed,
        initial_step_size=STEP_SIZE,
        extra=Dict{String,Any}("D" => D, "T" => T, "scale" => SCALE,
                               "max_tree_depth" => 5))

elseif MODE == "point"
    # Evaluate logpdf value at fixed z points (density parity vs NumPyro).
    zpts = [[0.0, 0.0], [0.3, -0.2], [-0.5, 0.1], [0.2, 0.5]]
    eval_lp(z, L, x0, y) = heat_inverse_logpdf(z, L, x0, y)
    println("  compiling point evaluator (optimize=:probprog, fresh x0 per call)...")
    # Fresh x0 per call: the compiled forward evaluator writes the final PDE
    # state back into the caller's x0 buffer (argument mutation write-back from
    # the in-model `F \ x` recurrence; donated_args=:none does not prevent it),
    # so reusing one x0 handle would silently chain the executions. The mcmc
    # path is unaffected (its compiled main writes back only the rng state).
    z_r0 = Reactant.to_rarray(zpts[1])
    x0_c = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
    compiled_lp = @compile compile_options = COMPILE_OPTS eval_lp(z_r0, L_r, x0_c, y_r)
    results = Dict{String,Float64}()
    for z in zpts
        z_r = Reactant.to_rarray(z)
        x0_fresh = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
        val = Float64(compiled_lp(z_r, L_r, x0_fresh, y_r))
        results["$(z)"] = val
        println("    logpdf(z=$z) = $(val)")
    end
    open(joinpath(dirname(data_path), "heat_logpdf_julia_scale$(SCALE).json"), "w") do io
        JSON3.write(io, results)
    end
    println("  wrote julia logpdf values.")

elseif MODE == "parity"
    # Dump NUTS samples with matched RNG/settings for lockstep parity vs NumPyro.
    num_samples = 20
    seed = UInt64[1, 5]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    println("  compiling parity run (arm $ARM)...")
    compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, L_r, x0_r, y_r; num_samples=num_samples)
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    # Thunks take no kwargs at call time; num_samples was baked at compile.
    samples, diagnostics, _, _ = compiled_fn(rng2, pos0, ss, imm_r, L_r, x0_r, y_r)
    samples_arr = Array(samples)          # num_samples x 2
    diag_arr = Array(diagnostics)
    println("  samples (z_kappa, z_sigma):")
    for j in 1:size(samples_arr, 1)
        println("    [$j] $(samples_arr[j, 1])  $(samples_arr[j, 2])   accepted=$(diag_arr[j])")
    end
    out = Dict("seed" => Int.(seed), "step_size" => STEP_SIZE, "max_tree_depth" => 5,
        "samples" => [collect(samples_arr[j, :]) for j in 1:size(samples_arr, 1)],
        "accepted" => Int.(diag_arr))
    open(joinpath(dirname(data_path), "heat_samples_julia_scale$(SCALE).json"), "w") do io
        JSON3.write(io, out)
    end
    println("  wrote julia samples. n_accepted=$(sum(diag_arr))")

elseif MODE == "ir"
    # Fast structural check: @code_hlo only, no timing (cheap even for arm A).
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, L_r, x0_r, y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    hlo_path = joinpath(dirname(data_path), "heat_hlo_arm$(ARM)_scale$(SCALE).mlir")
    open(io -> print(io, hlo_str), hlo_path, "w")
    println("  saved HLO: $hlo_path")

else  # bench
    println("  compiling (optimize=:probprog, disable_opt=$DISABLE_OPT, disable_sicm=$DISABLE_SICM, unroll=$UNROLL)...")
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos0, ss, imm_r, L_r, x0_r, y_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    # Warmup
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos0, ss, imm_r, L_r, x0_r, y_r)

    # Timed runs (reduce to 3 if a single run is slow; noted in report)
    n_runs = 5
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos0, ss, imm_r, L_r, x0_r, y_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    geo_t = exp(sum(log.(times)) / length(times))
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  Geomean (20 NUTS samples): $(round(geo_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    # Cross-arm verification run under the EXACT measured configuration (frozen
    # adapted state), so the arms-equivalence gate covers what was timed rather
    # than a separately-configured parity run.
    VSEED = UInt64[parse(UInt64, get(ENV, "PRELOWER_VERIFY_SEED", "7")), 11]
    rng_v = ProbProg.ReactantRNG(Reactant.to_rarray(copy(VSEED)))
    samples, diagnostics, log_densities, _, _ = compiled_fn(
        rng_v, pos0, ss, imm_r, L_r, x0_r, y_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println("  verification (seed $(Int.(VSEED))): n_accepted=$(sum(diag_arr[:, 1]))",
            " n_diverging=$(sum(diag_arr[:, 2]))")

    # IR evidence
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, L_r, x0_r, y_r)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)

    write_cell(joinpath(@__DIR__, "logs", "cells", "heat_scale$(SCALE)_arm$(ARM).json"),
        Dict(
            "model" => "heat_inverse", "size" => "D$(D)_T$(T)", "arm" => ARM,
            "D" => D, "T" => T, "M_obs" => M_OBS, "S" => S,
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
    # Save full HLO for scale 1 (and smoke) for later structural inspection.
    if SCALE in (0, 1)
        hlo_path = joinpath(dirname(data_path), "heat_hlo_arm$(ARM)_scale$(SCALE).mlir")
        open(io -> print(io, hlo_str), hlo_path, "w")
        println("  saved HLO: $hlo_path")
    end
end
