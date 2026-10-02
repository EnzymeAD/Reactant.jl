#!/usr/bin/env julia
#=
Cochlear-Aqueduct Diffusion (published model, real CT data)
============================================================

Public-provenance row of the pre-lowering ablation. Model AND data are those of

  A. M. A. Alghamdi, B. K. Mathiesen, L. Miyakoshi, M. Nedergaard, J. S. Jorgensen,
  P. A. R. Bork. Cochlear Aqueduct Advection and Diffusion Inferred from Computed
  Tomography Imaging with a Bayesian Approach. JARO 27(4):555-567, 2026.
  doi:10.1007/s10162-026-01047-x.  Code + data: github.com/CUQI-DTU/Paper-Cochlear,
  Zenodo doi:10.5281/zenodo.15796673 (v1.1.0, CC-BY-4.0).

Diffusion-only model of the paper (real-data configuration of
real_data_results.ipynb), mouse 1 right ear by default:

    dc/dt = d/dx( D(x) dc/dx ),   D = Ic * (d .^ 2),  d in R^21 (coarse nodes)
    backward Euler on n ~ 60 interior FD nodes:  A(D) u_{k+1} = u_k + dt_k b_k(D)
    A(D) = I + dt Dx' diag(D) Dx    (SPD, depends on the sample, constant per dt)
    5 steps of dt_s ~ 25.7 s, then 13 steps of dt_l ~ 128.6 s
    y = FD-gradient( spline(U) at ROI locations and scan times )    (4 x 7)
    y_obs ~ N(y, delta^{-1} I),  d ~ N(20, 5^2 I),  delta ~ Gamma(1.2, rate 5)

The published implementation (CUQIpy TimeDependentLinearPDE, backward_euler)
assembles the dense operator and solves INSIDE the time loop, which is exactly
what this program writes: the construction Dx' diag(D) Dx and the Cholesky are
in the loop body. Before AD, LICM hoists both out of each phase loop; after AD
every step holds a Cholesky pullback and a construction adjoint whose
cotangents depend on the state.

Fixed maps (Ic, Dx, boundary values, the separable spline observation factors
Px/Pt, the FD gradient) are exported from the SOURCE code by
cochlear_diffusion_export.py into cochlear_data/cochlear_<animal>_<ear>.json.

Sampled position (1 x 22): d[1:21], z = log(delta). Log density includes the
log-Jacobian of delta = exp(z); the posterior over (d, delta) is the paper's.

Arms (prelowering_common.jl): A = No Pre-Opt (flat, LICM off), B = Pre-Opt.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/cochlear_diffusion.jl <A|B> [mode]
  mode: bench (default) | warmup | ir | point | posterior
  env : COCHLEAR_CASE (default m1_r)
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "prelowering_common.jl"))

const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const MODE = length(ARGS) >= 2 ? ARGS[2] : "bench"
@assert ARM in ("A", "B") "arm must be A or B (got $ARM)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]
const CASE = get(ENV, "COCHLEAR_CASE", "m1_r")
const NUM_WARMUP = parse(Int, get(ENV, "PRELOWER_WARMUP", "200"))
const MAX_TREE_DEPTH = parse(Int, get(ENV, "COCHLEAR_MAX_DEPTH", "10"))  # source: max_depth 10
const STATE_PATH = get(
    ENV, "PRELOWER_STATE", joinpath(@__DIR__, "adapted_states", "cochlear_$(CASE).json")
)

# ── Data (exported from the source code) ─────────────────────────────────────
const DATA_PATH = joinpath(@__DIR__, "cochlear_data", "cochlear_$(CASE).json")
const RAW = JSON3.read(read(DATA_PATH, String))
_mat(rows) = permutedims(reduce(hcat, [collect(Float64, r) for r in rows]))
const NG = Int(RAW["n"])            # interior FD nodes
const NT = Int(RAW["nt"])           # time points incl. t = 0
const NC = Int(RAW["nc"])           # coarse diffusivity nodes (21)
const HH = Float64(RAW["h"])
const DTS = collect(Float64, RAW["dts"])
const N_SMALL = count(<(0.5 * maximum(DTS)), DTS)
const N_LARGE = NT - 1 - N_SMALL
# Phase step sizes. The source's tau array gives per-step differences that
# agree with these to a few ULP (documented in README; forward parity 1e-13).
const DT_S = DTS[1]
const DT_L = DTS[end]
@assert all(abs.(DTS[1:N_SMALL] .- DT_S) .< 1e-12 * DT_S)
@assert all(abs.(DTS[(N_SMALL + 1):end] .- DT_L) .< 1e-12 * DT_L)
const NP = NC + 1                   # position size
const NOBS = length(RAW["data"]) * length(RAW["data"][1])
const LOG2PI = log(2 * pi)
const PRIOR_MU = Float64(RAW["prior_d_mean"])
const PRIOR_SD = Float64(RAW["prior_d_std"])
const GA = Float64(RAW["gamma_shape"])
const GB = Float64(RAW["gamma_rate"])
const LGAMMA_GA = -0.08537409000331608   # lgamma(1.2), math.lgamma in CPython
@assert GA == 1.2

const IC = _mat(RAW["Ic"])                        # (NG+1) x NC
const DX = _mat(RAW["Dx"])                        # (NG+1) x NG
const DXT = permutedims(DX)                       # NG x (NG+1)
const GP = _mat(RAW["Gfd"]) * _mat(RAW["Px"])     # 4 x NG   (FD gradient of spline-in-space)
const PTT = permutedims(_mat(RAW["Pt"]))          # NT x 7   (spline-in-time weights)
const YOBS = _mat(RAW["data"])                    # 4 x 7
const BLR = hcat(collect(Float64, RAW["bl"]), collect(Float64, RAW["br"]))  # (NT-1) x 2
const U0 = reshape(collect(Float64, RAW["u0"]), NG, 1)
const E2 = let E = zeros(NG, 2)
    E[1, 1] = 1.0
    E[NG, 2] = 1.0
    E
end      # [e_1 e_n]
const SEL = let S = zeros(2, NG + 1)
    S[1, 1] = 1.0
    S[2, NG + 1] = 1.0
    S
end  # picks D_0, D_n

# ── Model: log joint density ─────────────────────────────────────────────────
function cochlear_logpdf(position, IC, DX, DXT, GP, PTT, YOBS, BLR, U0, E2, SEL)
    # The sampler passes a length-NP vector; linear indexing also accepts 1 x NP.
    dcoarse = reshape(position[1:NC], NC, 1)        # NC x 1
    z = @allowscalar position[NP]
    delta = exp(z)
    Dfine = IC * (dcoarse .* dcoarse)               # (NG+1) x 1, D on the fine grid
    Dbnd = SEL * Dfine                              # 2 x 1: D_0, D_n (boundary source)
    I_n = Matrix{Float64}(I, NG, NG)

    u = U0 .+ 0.0                                   # NG x 1 state, traced
    acc = (GP * u) * PTT[1:1, :]                    # 4 x 7 observation accumulator

    # Phase 1: the five short steps (dt_s). Operator constant across the phase.
    @trace track_numbers = false for t in 0:(N_SMALL - 1)
        M = I_n .+ DT_S .* (DXT * (Dfine .* DX))    # A(D), loop-invariant, O(n^3)
        F = cholesky(M)                             # loop-invariant factorization
        bnd = E2 * (Dbnd .* reshape(BLR[t + 1, :], 2, 1))
        u = F \ (u .+ (DT_S / (HH * HH)) .* bnd)    # 2 triangular solves
        acc = acc .+ (GP * u) * reshape(PTT[t + 2, :], 1, 7)
    end
    # Phase 2: the regular steps (dt_l).
    @trace track_numbers = false for t in 0:(N_LARGE - 1)
        M = I_n .+ DT_L .* (DXT * (Dfine .* DX))
        F = cholesky(M)
        bnd = E2 * (Dbnd .* reshape(BLR[t + 1 + N_SMALL, :], 2, 1))
        u = F \ (u .+ (DT_L / (HH * HH)) .* bnd)
        acc = acc .+ (GP * u) * reshape(PTT[t + 2 + N_SMALL, :], 1, 7)
    end

    r = YOBS .- acc
    loglik = 0.5 * NOBS * (z - LOG2PI) - 0.5 * delta * sum(r .* r)
    dz = (dcoarse .- PRIOR_MU) ./ PRIOR_SD
    logprior_d = -0.5 * sum(dz .* dz) - NC * (log(PRIOR_SD) + 0.5 * LOG2PI)
    # delta ~ Gamma(shape GA, rate GB), plus log|d delta / dz| = z
    logprior_z = GA * log(GB) - LGAMMA_GA + (GA - 1) * z - GB * delta + z
    return loglik + logprior_d + logprior_z
end

const MARGS_HOST = (IC, DX, DXT, GP, PTT, YOBS, BLR, U0, E2, SEL)

function run_mcmc(rng, pos, ss, imm, margs...; num_samples=20)
    return ProbProg.mcmc_logpdf(
        rng,
        cochlear_logpdf,
        pos,
        margs...;
        algorithm=:NUTS,
        step_size=ss,
        inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH,
        num_warmup=0,
        num_samples=num_samples,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
end

function run_warmup(rng, pos, ss, imm, margs...)
    return ProbProg.mcmc_logpdf(
        rng,
        cochlear_logpdf,
        pos,
        margs...;
        algorithm=:NUTS,
        step_size=ss,
        inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH,
        num_warmup=NUM_WARMUP,
        num_samples=1,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
end

const POST_WARMUP = parse(Int, get(ENV, "COCHLEAR_POST_WARMUP", "1000"))
const POST_SAMPLES = parse(Int, get(ENV, "COCHLEAR_POST_SAMPLES", "2000"))
function run_posterior(rng, pos, ss, imm, margs...)
    return ProbProg.mcmc_logpdf(
        rng,
        cochlear_logpdf,
        pos,
        margs...;
        algorithm=:NUTS,
        step_size=ss,
        inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH,
        num_warmup=POST_WARMUP,
        num_samples=POST_SAMPLES,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
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

println(
    "=== Cochlear diffusion ($CASE): arm $ARM (flat, licm ",
    ARM_DISABLE_LICM ? "OFF" : "ON",
    ", SICM OFF), n=$NG steps=$(NT-1) ",
    "($N_SMALL x $(round(DT_S, digits=4)) s + $N_LARGE x $(round(DT_L, digits=4)) s), ",
    "P=$NP, max_tree_depth=$MAX_TREE_DEPTH, mode=$MODE ===",
)

margs = map(Reactant.to_rarray, MARGS_HOST)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, NP, NP))
# Source initial point: d = 20 (sample_the_posterior); delta at its prior mean.
const POS_INIT = reshape(vcat(fill(20.0, NC), log(GA / GB)), 1, NP)
pos0 = Reactant.to_rarray(POS_INIT)
const STEP_SIZE = parse(Float64, get(ENV, "COCHLEAR_STEP", "0.1"))   # source NUTS step 0.1
ss = Reactant.ConcreteRNumber(STEP_SIZE)

if MODE == "bench"
    ST = load_adapted_state(STATE_PATH)
    ss = Reactant.ConcreteRNumber(ST.step)
    imm_r = Reactant.to_rarray(ST.imm)
    pos0 = Reactant.to_rarray(ST.pos)
    println("  frozen-adapted state: $(STATE_PATH)")
    println(
        "    step=$(ST.step)  inv_mass_rank=$(ndims(ST.imm))  ",
        "warmup=$(ST.meta["num_warmup"]) by arm $(ST.meta["produced_by_arm"])",
    )
end

if MODE == "warmup"
    @assert ARM == "B" "warmup must run in the Pre-Opt arm (B); got $ARM"
    seed = UInt64[42, 0]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    ct = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_warmup(
            rng, pos0, ss, imm_r, margs...
        )
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed ((_, _, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, margs...))
    println("  Warmup run: $(round(wt, digits=1))s")
    save_adapted_state(
        STATE_PATH,
        "cochlear_$(CASE)",
        extract_adapted_state(st);
        num_warmup=NUM_WARMUP,
        produced_by_arm=ARM,
        warmup_seed=seed,
        initial_step_size=STEP_SIZE,
        extra=Dict{String,Any}(
            "n" => NG,
            "steps" => NT - 1,
            "case" => CASE,
            "max_tree_depth" => MAX_TREE_DEPTH,
            "data" => DATA_PATH,
        ),
    )

elseif MODE == "point"
    # Log density at the source's reference points (parity with CUQIpy logd).
    lp(p, a...) = cochlear_logpdf(p, a...)
    p_r0 = Reactant.to_rarray(vec(POS_INIT))
    compiled_lp = @compile compile_options = COMPILE_OPTS lp(p_r0, margs...)
    out = Any[]
    for rp in RAW["reference_points"]
        p = vcat(collect(Float64, rp["d"]), log(Float64(rp["s"])))
        v = Float64(
            compiled_lp(Reactant.to_rarray(p), map(Reactant.to_rarray, MARGS_HOST)...)
        )
        v_nojac = v - log(Float64(rp["s"]))     # source logd has no log-Jacobian
        d = v_nojac - Float64(rp["joint_logd"])
        println("    logd impl $(v_nojac)  source $(rp["joint_logd"])  diff $(d)")
        push!(out, Dict("impl" => v_nojac, "source" => rp["joint_logd"], "diff" => d))
    end
    write_cell(
        joinpath(@__DIR__, "logs", "cells", "cochlear_$(CASE)_point_arm$(ARM).json"),
        Dict("points" => out),
    )

elseif MODE == "posterior"
    # Posterior validation run (Pre-Opt arm): adaptation on, then samples.
    seed = UInt64[parse(UInt64, get(ENV, "COCHLEAR_SEED", "2024")), 0]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    ct = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_posterior(
            rng, pos0, ss, imm_r, margs...
        )
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed (
        (samples, diagnostics, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, margs...)
    )
    S = Array(samples)
    Dg = Array(diagnostics)
    println(
        "  Run: $(round(wt, digits=1))s  samples=$(size(S))  accept=$(sum(Dg[:, 1]))  div=$(sum(Dg[:, 2]))",
    )
    Dsq = S[:, 1:NC] .^ 2
    Dbar = sum(sum(Dsq; dims=1) ./ size(S, 1)) / NC
    sig = mean_sig = sum(exp.(-0.5 .* S[:, NP])) / size(S, 1)
    println("  posterior D_bar (mean over coarse nodes of E[d^2]) = $(Dbar) um^2/s")
    println("  posterior mean sigma_noise = $(sig)")
    write_cell(
        joinpath(
            @__DIR__, "logs", "cells", "cochlear_$(CASE)_posterior_seed$(seed[1]).json"
        ),
        Dict(
            "seed" => Int.(seed),
            "num_warmup" => POST_WARMUP,
            "num_samples" => POST_SAMPLES,
            "max_tree_depth" => MAX_TREE_DEPTH,
            "run_s" => wt,
            "samples" => [collect(S[j, :]) for j in 1:size(S, 1)],
            "accepted" => Int.(Dg[:, 1]),
            "diverging" => Int.(Dg[:, 2]),
            "adapted_step_size" => Array(st.step_size)[],
            "D_bar" => Dbar,
            "sigma_noise_mean" => sig,
        ),
    )

elseif MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos0, ss, imm_r, margs...)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "cochlear_$(CASE)_arm$(ARM).mlir"), hlo_str)

else  # bench
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos0, ss, imm_r, margs...
        )
    end
    println("  Compile: $(round(compile_time, digits=1))s")
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos0, ss, imm_r, margs...)

    n_runs = 5
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos0, ss, imm_r, margs...)
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
    samples, diagnostics, log_densities, _, _ = compiled_fn(
        rng_v, pos0, ss, imm_r, margs...
    )
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println(
        "  verification (seed $(Int.(VSEED))): n_accepted=$(sum(diag_arr[:, 1]))",
        " n_diverging=$(sum(diag_arr[:, 2]))",
    )

    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos0, ss, imm_r, margs...)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "cochlear_$(CASE)_arm$(ARM).mlir"), hlo_str)

    write_cell(
        joinpath(@__DIR__, "logs", "cells", "cochlear_$(CASE)_arm$(ARM).json"),
        Dict(
            "model" => "cochlear_diffusion",
            "case" => CASE,
            "arm" => ARM,
            "n" => NG,
            "steps" => NT - 1,
            "P" => NP,
            "max_tree_depth" => MAX_TREE_DEPTH,
            "num_samples" => 20,
            "compile_s" => compile_time,
            "times_ms" => times .* 1000,
            "median_ms" => median_t * 1000,
            "verify_seed" => Int.(VSEED),
            "samples" => [collect(samples_arr[j, :]) for j in 1:size(samples_arr, 1)],
            "accepted" => Int.(diag_arr[:, 1]),
            "diverging" => Int.(diag_arr[:, 2]),
            "log_densities" => collect(ld_arr),
            "ir" => counts,
        ),
    )
end
