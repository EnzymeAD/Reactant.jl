#!/usr/bin/env julia
#=
Cubic-Cost Kernel Dynamics (LICM with O(D³) loop-invariant construction)
=========================================================================

Model: Same recurrence as forward_model but with a more expensive kernel:
    x_{t+1} = K(θ) · x_t

where K(θ) = (θ · W)' * (θ · W) + I = θ² · W'W + I.

The difference from forward_model: we do NOT precompute W'W. Instead, the
matrix-matrix product (θ·W)' * (θ·W) is computed INSIDE the loop.
This is an O(D³) operation (vs O(D²) element-wise scaling in forward_model).

Pre-AD LICM hoists the O(D³) matrix multiply outside the loop. Without LICM,
the backward pass pays O(T · D³) for the matmul adjoint per iteration.

Arm mechanism (2026-08-24): unified with the heat rows of tab:prelowering
(heat_inverse.jl / heat_inverse_RESULTS_2026-08-21.md). Both arms run the flat
expand path (`DEBUG_PROBPROG_DISABLE_OPT=true`) with SICM explicitly off, and
differ ONLY in `disable_licm_optimization_passes` (CompileOptions; strips the
`*_licm` pattern subset from the standard passes, pre- and post-AD). The old
`optimize=:probprog_no_opt` arm definition (region-vs-flat) is gone from
Reactant and is NOT what this table measures.

Usage (from Reactant.jl root):
    JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
        benchmark/probprog/ablation/pre_lowering/cubic_kernel_dynamics.jl \
        <A|B> <small|large> [bench|ir]
    small = D=60,  T=50   ("Cubic Kernel" row)
    large = D=100, T=100  ("Cubic Kernel (large)" row)
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg

include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const SIZE = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "small"
const MODE = length(ARGS) >= 3 ? ARGS[3] : "bench"
@assert ARM in ("A", "B") "arm must be A or B (got $ARM)"
@assert SIZE in ("small", "large") "size must be small or large (got $SIZE)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]

const D_STATE = SIZE == "large" ? 100 : 60   # state dimension (matmul cost = O(D³))
const T_STEPS = SIZE == "large" ? 100 : 50   # propagation steps

function cubic_kernel_logpdf(position, W_basis, x0)
    @allowscalar begin
        theta = position[1, 1]
        I_d = Matrix{Float64}(I, D_STATE, D_STATE)

        x = x0   # 1 × D
        @trace track_numbers=false for t in 0:(T_STEPS-1)
            # Kernel construction: O(D³) matrix-matrix multiply (LOOP-INVARIANT)
            W_scaled = theta .* W_basis                        # D × D, O(D²)
            K = permutedims(W_scaled) * W_scaled .+ I_d        # D × D, O(D³)
            # State update (loop-variant recurrence)
            x = x * K                                          # 1 × D, O(D²)
        end

        total = -0.5 * sum(x .* x)
        return total
    end
end

function run_mcmc(rng, pos, ss, W, x0)
    ProbProg.mcmc_logpdf(rng, cubic_kernel_logpdf, pos, W, x0;
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

println("=== Cubic Kernel Dynamics: arm $ARM (path ",
        ARM_DISABLE_OPT ? "flat" : "region",
        ", licm ", ARM_DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", ARM_DISABLE_SICM ? "OFF" : "ON",
        "), size $SIZE, D=$D_STATE T=$T_STEPS, mode=$MODE ===")

Random.seed!(42)
W_r = Reactant.to_rarray(randn(D_STATE, D_STATE) * 0.01)
x0_r = Reactant.to_rarray(ones(1, D_STATE) * 0.01)
pos = Reactant.to_rarray(ones(1, 1) * 0.5)
ss = Reactant.ConcreteRNumber(0.1)
rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))

if MODE == "ir"
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos, ss, W_r, x0_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
else  # bench
    compile_time = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
            rng, pos, ss, W_r, x0_r)
    end
    println("  Compile: $(round(compile_time, digits=1))s")

    # Warmup
    rng1 = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    compiled_fn(rng1, pos, ss, W_r, x0_r)

    # Timed runs (protocol: 5; env override is for jitter diagnostics only)
    n_runs = parse(Int, get(ENV, "PRELOWER_NRUNS", "5"))
    times = Float64[]
    for i in 1:n_runs
        rng_i = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[100 + i, 0]))
        t = @elapsed compiled_fn(rng_i, pos, ss, W_r, x0_r)
        push!(times, t)
    end
    sorted_t = sort(times)
    median_t = sorted_t[(length(sorted_t) + 1) ÷ 2]
    println("  Median (20 NUTS samples): $(round(median_t * 1000, digits=1))ms")
    println("  All: $(round.(times .* 1000, digits=1))ms")

    # Cross-arm verification run: fixed seed, dump samples + accept decisions.
    rng_v = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[7, 11]))
    samples, diagnostics, log_densities, _, _ = compiled_fn(rng_v, pos, ss, W_r, x0_r)
    samples_arr = Array(samples)
    diag_arr = Array(diagnostics)
    ld_arr = Array(log_densities)
    println("  verification (seed [7,11]): n_accepted=$(sum(diag_arr[:, 1]))")

    # IR evidence
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(rng, pos, ss, W_r, x0_r)
    hlo_str = sprint(show, hlo)
    counts = ir_report(hlo_str)
    save_hlo(joinpath(@__DIR__, "logs", "hlo", "cubic_$(SIZE)_arm$(ARM).mlir"), hlo_str)

    write_cell(joinpath(@__DIR__, "logs", "cells", "cubic_$(SIZE)_arm$(ARM).json"),
        Dict(
            "model" => "cubic_kernel", "size" => SIZE, "arm" => ARM,
            "D" => D_STATE, "T" => T_STEPS,
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
