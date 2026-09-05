#!/usr/bin/env julia
#=
Advection-Diffusion Contaminant Transport: Bayesian Inverse Problem
===================================================================

Second physics for the in-loop-factorization mechanism of tab:prelowering.
Infer log-diffusivity, log-velocity and log-noise from noisy sensor readings of
an IMEX (implicit diffusion / explicit upwind advection) transport solve:

    kappa = exp(z_kappa),  v = exp(z_v),  sigma = exp(z_sigma)
    M(kappa) = I + dt*kappa*L                        (SPD, loop-invariant)
    M(kappa) x_{t+1} = x_t - dt*v*(G x_t)            (one chol + 2 solves/step)
    y_j ~ Normal(H x_{t_j}, sigma^2 I)

The Cholesky of M(kappa) is loop-invariant across the T time steps but is kept
INSIDE the traced loop in source, exactly as in heat_inverse.jl. The explicit
advection flux G x_t is loop-VARIANT and must stay in the loop; that is the
point of the IMEX split, and it makes this a stricter test of the LICM pattern
than heat (the loop body is no longer purely invariant plus a solve).

Provenance framing (same rules as the heat rows): the problem class is standard
Bayesian parameter identification for contaminant transport, posed in the
standard framework for Bayesian inverse problems (Stuart 2010) with the
inverse-crime protocol of Kaipio & Somersalo (2005). The INSTANCE (grid,
sensors, priors, scales) is constructed for this paper and must be presented as
"a representative Bayesian inverse problem of the standard class", never as an
existing published benchmark.

Arms: the unified tab:prelowering two-arm definition (see
prelowering_common.jl and heat_inverse_RESULTS_2026-08-21.md). Both arms run
the flat expand path with SICM explicitly off, and differ ONLY in
`disable_licm_optimization_passes`.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/advection_diffusion.jl <A|B> <scale> [mode]
  scale: 0 (smoke D=20,T=10), 1 (D=200,T=100 table row), 2 (D=100,T=100)
  mode : bench (default) | ir
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

include(joinpath(@__DIR__, "advection_diffusion_data.jl"))
include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const SCALE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const MODE = length(ARGS) >= 3 ? ARGS[3] : "bench"
@assert ARM in ("A", "B") "arm must be A or B (got $ARM)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]
const NUM_WARMUP = parse(Int, get(ENV, "PRELOWER_WARMUP", "200"))
const STATE_PATH = get(ENV, "PRELOWER_STATE",
    joinpath(@__DIR__, "adapted_states", "advdiff_scale$(SCALE).json"))

# ── Config / constants (baked for the traced logpdf) ─────────────────────────
const CFG = advdiff_config(SCALE)
const D = CFG.D
const T = CFG.T
const DT = CFG.dt
const M_OBS = CFG.M_obs
const S = CFG.S
const OBS_STEP = CFG.obs_step
const SENSOR_IDX = CFG.sensor_idx
const LOG2PI = log(2 * pi)
const STAB = advdiff_stability(CFG)

# ── Model: log joint density ─────────────────────────────────────────────────
#
# Chunked form matching heat_inverse.jl: M_obs unrolled observation windows,
# each a traced 0-based for loop advancing OBS_STEP IMEX steps with the
# cholesky INSIDE the loop body. Likelihood accumulated between windows.
function advdiff_logpdf(position, L, G, x0col, y_data)
    z_kappa = @allowscalar position[1]
    z_v = @allowscalar position[2]
    z_sigma = @allowscalar position[3]
    kappa = exp(z_kappa)
    v = exp(z_v)
    sigma = exp(z_sigma)

    I_d = Matrix{Float64}(I, D, D)
    xcol = x0col                              # D x 1 state at coarse step 0

    loglik = -1.0 * (M_OBS * S) * (0.5 * LOG2PI + z_sigma)
    for j in 1:M_OBS
        @trace track_numbers = false for t in 0:(OBS_STEP - 1)
            # Implicit diffusion operator: loop-invariant SPD matrix + factor.
            M = I_d .+ (DT * kappa) .* L        # D x D, SPD, loop-invariant
            F = cholesky(M)                      # O(D^3), loop-invariant
            # Explicit upwind advection flux: loop-VARIANT, must stay in-loop.
            rhs = xcol .- (DT * v) .* (G * xcol) # D x 1
            xcol = F \ rhs                       # 2 triangular solves
        end
        obs_j = xcol[SENSOR_IDX, 1]
        resid = y_data[j, :] .- obs_j
        loglik = loglik - 0.5 * sum(resid .* resid) / (sigma * sigma)
    end

    logprior = -0.5 * (z_kappa * z_kappa + z_v * z_v + z_sigma * z_sigma) -
               1.5 * LOG2PI
    return logprior + loglik
end

function run_mcmc(rng, pos, ss, imm, L, G, x0col, y_data; num_samples=20)
    ProbProg.mcmc_logpdf(rng, advdiff_logpdf, pos, L, G, x0col, y_data;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# Warmup phase of the frozen-adapted protocol: adaptation ON, not timed.
function run_warmup(rng, pos, ss, imm, L, G, x0col, y_data)
    ProbProg.mcmc_logpdf(rng, advdiff_logpdf, pos, L, G, x0col, y_data;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
        num_warmup=NUM_WARMUP, num_samples=1,
        adapt_step_size=true, adapt_mass_matrix=true)
end

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = ARM_DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = ARM_DISABLE_SICM
# NOTE: @compile ignores its other kwargs (incl. `sync`) when `compile_options`
# is given, so sync must be set HERE or timings measure only async dispatch.
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=ARM_DISABLE_LICM,
    sync=true,
)

println("=== Advection-Diffusion: arm $ARM (path ",
        ARM_DISABLE_OPT ? "flat" : "region",
        ", licm ", ARM_DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", ARM_DISABLE_SICM ? "OFF" : "ON",
        "), scale $SCALE, D=$D T=$T M_obs=$M_OBS S=$S, mode=$MODE ===")
println("  truth: kappa=$(CFG.kappa_true) v=$(CFG.v_true) sigma=$(CFG.sigma_true) ",
        "dt=$(round(DT, sigdigits=4)) h=$(round(STAB.h, sigdigits=4))")
println("  stability: dt / explicit-diffusion-limit = ",
        round(STAB.stiffness, digits=1), "x (implicit stepping required); ",
        "advection CFL = v*dt/h = ", round(STAB.cfl, digits=4), " (< 1, explicit OK); ",
        "Peclet = v*L/kappa = ", round(STAB.peclet, digits=1))
println("  fine (data) grid: stiffness ", round(STAB.stiffness_fine, digits=1),
        "x, CFL ", round(STAB.cfl_fine, digits=4))

cfg_data = advdiff_generate_data(CFG)
data_path = advdiff_data_path(SCALE)
advdiff_save_data(data_path, CFG, cfg_data)
println("  data file: $data_path")

L_r = Reactant.to_rarray(advdiff_laplacian(D))
G_r = Reactant.to_rarray(advdiff_upwind(D))
x0_r = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
y_r = Reactant.to_rarray(cfg_data.y)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, 3, 3))     # dense identity mass
pos0 = Reactant.to_rarray(reshape([0.0, 0.0, 0.0], 1, 3))
const STEP_SIZE = parse(Float64, get(ENV, "ADVDIFF_STEP", "0.02"))
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
            rng, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed ((_, _, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, L_r, G_r, x0_r, y_r))
    println("  Warmup run: $(round(wt, digits=1))s")
    save_adapted_state(STATE_PATH, "advdiff_scale$(SCALE)", extract_adapted_state(st);
        num_warmup=NUM_WARMUP, produced_by_arm=ARM, warmup_seed=seed,
        initial_step_size=STEP_SIZE,
        extra=Dict{String,Any}("D" => D, "T" => T, "scale" => SCALE,
                               "max_tree_depth" => 5))

elseif MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo",
                      "advdiff_scale$(SCALE)_arm$(ARM).mlir"), hlo_str)
else  # bench
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    # Warmup
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)

    # Timed runs (protocol: 5; env override is for jitter diagnostics only)
    n_runs = parse(Int, get(ENV, "PRELOWER_NRUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    geo_t = exp(sum(log.(times)) / length(times))
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  Geomean (20 NUTS samples): $(round(geo_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    # Cross-arm verification run: fixed seed, dump samples + accept decisions.
    VSEED = UInt64[parse(UInt64, get(ENV, "PRELOWER_VERIFY_SEED", "7")), 11]
    rng_v = ProbProg.ReactantRNG(Reactant.to_rarray(copy(VSEED)))
    samples, diagnostics, log_densities, _, _ = compiled_fn(
        rng_v, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println("  verification (seed $(Int.(VSEED))): n_accepted=$(sum(diag_arr[:, 1]))",
            " n_diverging=$(sum(diag_arr[:, 2]))")

    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, L_r, G_r, x0_r, y_r)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo",
                      "advdiff_scale$(SCALE)_arm$(ARM).mlir"), hlo_str)

    write_cell(joinpath(@__DIR__, "logs", "cells",
                        "advdiff_scale$(SCALE)_arm$(ARM).json"),
        Dict(
            "model" => "advection_diffusion", "size" => "D$(D)_T$(T)", "arm" => ARM,
            "D" => D, "T" => T, "M_obs" => M_OBS, "S" => S,
            "step_size" => STEP_SIZE, "max_tree_depth" => 5, "num_samples" => 20,
            "kappa_true" => CFG.kappa_true, "v_true" => CFG.v_true,
            "sigma_true" => CFG.sigma_true, "dt" => DT, "h" => STAB.h,
            "stiffness_ratio" => STAB.stiffness, "advection_cfl" => STAB.cfl,
            "peclet" => STAB.peclet,
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
