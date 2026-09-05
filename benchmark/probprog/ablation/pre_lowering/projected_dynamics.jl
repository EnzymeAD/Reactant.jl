#!/usr/bin/env julia
#=
Projected Dynamics with Two Loop-Invariant Matrices
=====================================================

Model: A D-dimensional latent state evolves, observed through a projection:
    x_{t+1} = K(θ) · x_t                (transition, K loop-invariant)
    z_T = P(θ) · x_T                    (final observation through P)
    log p(y | θ) = -½ ||z_T - y||²

where K(θ) = θ · A + I and P(θ) = θ · B' * B + I_obs.

Both K(θ) and P(θ) are loop-invariant. LICM hoists BOTH matrix constructions
outside the loop before AD, reducing the backward pass cost for both.

Comparison with forward_model.jl:
  forward_model: 1 invariant matrix, D²
  this model:    2 invariant matrices (D² transition + D_obs×D projection),
                 demonstrates LICM benefit scales with number of invariant ops.

Arm mechanism (2026-08-24): unified with the heat rows of tab:prelowering
(heat_inverse.jl / heat_inverse_RESULTS_2026-08-21.md). Both arms run the flat
expand path (`DEBUG_PROBPROG_DISABLE_OPT=true`) with SICM explicitly off, and
differ ONLY in `disable_licm_optimization_passes` (CompileOptions; strips the
`*_licm` pattern subset from the standard passes, pre- and post-AD). The old
`optimize=:probprog_no_opt` arm definition (region-vs-flat) is gone from
Reactant and is NOT what this table measures.

Usage (from Reactant.jl root):
    JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/projected_dynamics.jl \
        <A|B> [bench|ir]
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const MODE = length(ARGS) >= 2 ? ARGS[2] : "bench"
@assert ARM in ("A", "B") "arm must be A or B (got $ARM)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]

const D_STATE = 100    # state dimension
const D_OBS = 30       # observation dimension
const T_STEPS = 100    # propagation steps

function projected_dynamics_logpdf(position, A_basis, B_basis, y_target, x0)
    @allowscalar begin
        theta = position[1, 1]
        I_d = Matrix{Float64}(I, D_STATE, D_STATE)
        I_obs = Matrix{Float64}(I, D_OBS, D_OBS)

        x = x0   # 1 × D_STATE
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Transition matrix (LOOP-INVARIANT #1)
            K = theta .* A_basis .+ I_d                        # D × D
            # State update
            x = x * K                                          # 1 × D
        end

        # Observation projection (INVARIANT #2: depends on θ, computed once)
        # P(θ) = θ · B'B + I_obs
        P = theta .* (permutedims(B_basis) * B_basis) .+ I_obs  # D_OBS × D_OBS

        # Project final state: z = x_T · B · P^{-1}... simplified:
        # Just use P as a mixing matrix on the observation side
        z = x[:, 1:D_OBS] * P                                 # 1 × D_OBS

        diff = z .- y_target
        total = -0.5 * sum(diff .* diff)
        return total
    end
end

function run_mcmc(rng, pos, ss, A_basis, B_basis, y_target, x0)
    ProbProg.mcmc_logpdf(rng, projected_dynamics_logpdf, pos, A_basis, B_basis, y_target, x0;
        algorithm=:NUTS, step_size=ss, max_tree_depth=5,
        num_warmup=0, num_samples=20,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = ARM_DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = ARM_DISABLE_SICM
# NOTE: the @compile macro ignores its other kwargs (incl. `sync`) when
# `compile_options` is given, so sync must be set HERE or timings measure
# only async dispatch.
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=ARM_DISABLE_LICM,
    sync=true,
)

println("=== Projected Dynamics: arm $ARM (path ",
        ARM_DISABLE_OPT ? "flat" : "region",
        ", licm ", ARM_DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", ARM_DISABLE_SICM ? "OFF" : "ON",
        "), D=$D_STATE D_obs=$D_OBS T=$T_STEPS, mode=$MODE ===")

Random.seed!(42)
A_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.01)
B_r = Reactant.to_rarray(randn(D_OBS, D_OBS) * 0.1)
y_r = Reactant.to_rarray(randn(1, D_OBS) * 0.1)
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

if MODE == "ir"
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos, ss, A_r, B_r, y_r, x0_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
else  # bench
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos, ss, A_r, B_r, y_r, x0_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    # Warmup
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos, ss, A_r, B_r, y_r, x0_r)

    # Timed runs (protocol: 5; env override is for jitter diagnostics only)
    n_runs = parse(Int, get(ENV, "PRELOWER_NRUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos, ss, A_r, B_r, y_r, x0_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    # Cross-arm verification run: fixed seed, dump samples + accept decisions.
    rng_v = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[7, 11]))
    samples, diagnostics, log_densities, _, _ = compiled_fn(
        rng_v, pos, ss, A_r, B_r, y_r, x0_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println("  verification (seed [7,11]): n_accepted=$(sum(diag_arr[:, 1]))")

    # IR evidence
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos, ss, A_r, B_r, y_r, x0_r)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "projected_arm$(ARM).mlir"), hlo_str)

    write_cell(joinpath(@__DIR__, "logs", "cells", "projected_armlarge_arm$(ARM).json"),
        Dict(
            "model" => "projected_dynamics", "size" => "D100_T100", "arm" => ARM,
            "D" => D_STATE, "D_obs" => D_OBS, "T" => T_STEPS,
            "step_size" => 0.1, "max_tree_depth" => 5, "num_samples" => 20,
            "compile_s" => compile_time,
            "times_ms" => times .* 1000, "median_ms" => median_t * 1000,
            "verify_seed" => [7, 11],
            "samples" => [collect(samples_arr[j, :]) for j in 1:size(samples_arr, 1)],
            "accepted" => Int.(diag_arr[:, 1]),
            "diverging" => Int.(diag_arr[:, 2]),
            "log_densities" => collect(ld_arr),
            "ir" => counts,
        ))
end
