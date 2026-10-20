#!/usr/bin/env julia
#=
IDE Radar (published model, real radar data), pre-lowering ablation row
=======================================================================

Model, data and spelling: see ide_radar_model.jl. Mechanism: the IDE propagator
M(theta) = GRAM^-1 PHI_g' K(theta) PHI_g area^2 (a 1681 x 1681 kernel matrix on the
quadrature grid, projected on 129 basis functions) depends on the sampled kernel parameters and
is loop-invariant in the Kalman-filter time loop. Before AD, LICM hoists its construction out
of the loop; after AD, every step of the reverse loop of the un-hoisted program carries the
construction adjoint (dK = PHI_g dY PHI_g', an O(NG^2 r) product, then the kernel pullback),
whose cotangent changes from step to step.

Arms (prelowering_common.jl): A = old "without" (flat, LICM off everywhere), B = "with"
(flat, full standard passes). The fair "without" arm N (LICM off before lowering only) is set
with Reactant.Compiler.DEBUG_PROBPROG_DISABLE_PRELOWER_LICM, see fig9a_paired_ide_radar.jl.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=<env> \
    benchmark/probprog/ablation/pre_lowering/ide_radar.jl <A|N|B> [mode]
  mode: point (default) | warmup | ir
  env : IDE_GRID (41 default, or 21), IDE_MAX_DEPTH (10), PRELOWER_WARMUP (200),
        IDE_OUT (output directory, default logs/), PRELOWER_STATE (adapted-state path),
        IDE_INIT_IMM (warmup start: JSON with "inverse_mass" and "mode"), IDE_STEP (0.01)
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "prelowering_common.jl"))
include(joinpath(@__DIR__, "ide_radar_model.jl"))

const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const MODE = length(ARGS) >= 2 ? ARGS[2] : "point"
@assert ARM in ("A", "N", "B") "arm must be A, N or B (got $ARM)"
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM == "N" ? "B" : ARM]
const NUM_WARMUP = parse(Int, get(ENV, "PRELOWER_WARMUP", "200"))
const MAX_TREE_DEPTH = parse(Int, get(ENV, "IDE_MAX_DEPTH", "10"))
const OUT_DIR = get(ENV, "IDE_OUT", joinpath(@__DIR__, "logs"))
const STATE_PATH = get(ENV, "PRELOWER_STATE",
    joinpath(@__DIR__, "adapted_states", "ide_radar_g$(IDE_GRID).json"))

function run_mcmc(rng, pos, ss, imm, margs...; num_samples=20)
    return ProbProg.mcmc_logpdf(rng, ide_logpdf, pos, margs...;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH, num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

function run_warmup(rng, pos, ss, imm, margs...)
    return ProbProg.mcmc_logpdf(rng, ide_logpdf, pos, margs...;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH, num_warmup=NUM_WARMUP, num_samples=1,
        adapt_step_size=true, adapt_mass_matrix=true)
end

Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = ARM_DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = ARM_DISABLE_SICM
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=ARM_DISABLE_LICM,
    sync=true,
)
# The fair "without" knob is read when the pipeline is built (inside @compile / @code_hlo).
with_arm(f) = begin
    Reactant.Compiler.DEBUG_PROBPROG_DISABLE_PRELOWER_LICM[] = ARM == "N"
    try
        f()
    finally
        Reactant.Compiler.DEBUG_PROBPROG_DISABLE_PRELOWER_LICM[] = false
    end
end

println("=== IDE radar: arm $ARM (flat, licm ", ARM_DISABLE_LICM ? "OFF" :
        ARM == "N" ? "OFF before lowering only" : "ON", ", SICM OFF), grid $(IDE_GRID)x$(IDE_GRID) ",
        "(NG=$NG), r=$R, steps=$NT, pixels/image=$NOBS, P=$NP, ",
        "max_tree_depth=$MAX_TREE_DEPTH, mode=$MODE ===")

margs = map(Reactant.to_rarray, MARGS_HOST)
imm_r = Reactant.to_rarray(Matrix{Float64}(I, NP, NP))
const POS_INIT = ide_fit_position()
pos0 = Reactant.to_rarray(POS_INIT)
# Optional warmup start (IDE_INIT_IMM = JSON with "inverse_mass" rows and "mode"), e.g. the
# Laplace approximation of ide_radar_jax_gradient.py's program; it only shortens warmup.
if haskey(ENV, "IDE_INIT_IMM")
    let L = JSON3.read(read(ENV["IDE_INIT_IMM"], String))
        global imm_r = Reactant.to_rarray(_rows_to_matrix(L["inverse_mass"]))
        global pos0 = Reactant.to_rarray(reshape(collect(Float64, L["mode"]), 1, NP))
        println("  warmup start from $(ENV["IDE_INIT_IMM"]) (dense inverse mass, mode)")
    end
end
const STEP_SIZE = parse(Float64, get(ENV, "IDE_STEP", "0.01"))
ss = Reactant.ConcreteRNumber(STEP_SIZE)

if MODE == "point"
    # Log joint at the reference points of export_ide_radar.R. The R side evaluates the IDE
    # package's own likelihood (joint sparse precision, construct_Q + negloglik formula);
    # the prior is added here in plain Julia from the same constants.
    lp(p, a...) = ide_logpdf(p, a...)
    compiled_lp = with_arm(() -> @compile compile_options = COMPILE_OPTS lp(
        Reactant.to_rarray(vec(POS_INIT)), margs...))
    out = Any[]
    for rp in IDE_RAW["reference_points"]
        k = collect(Float64, rp["k"])
        th = [log(k[1]), log(k[2]), k[3], k[4], log(Float64(rp["s2eps"])),
              log(Float64(rp["s2eta"])), Float64(rp["beta"])]
        v = Float64(compiled_lp(Reactant.to_rarray(th), map(Reactant.to_rarray, MARGS_HOST)...))
        zp = (th .- PRIOR_MU) ./ PRIOR_SD
        ref = (IDE_GRID == G41 ? Float64(rp["loglik"]) : NaN) - 0.5 * sum(zp .^ 2) + LOGPRIOR_CONST
        println("    log joint impl $(v)  reference (R likelihood + prior) $(ref)  diff $(v - ref)")
        push!(out, Dict("theta" => th, "impl" => v, "reference" => ref, "diff" => v - ref))
    end
    write_cell(joinpath(OUT_DIR, "cells", "ide_radar_g$(IDE_GRID)_point_arm$(ARM).json"),
        Dict("points" => out))

elseif MODE == "warmup"
    @assert ARM == "B" "warmup must run in the Pre-Opt arm (B); got $ARM"
    seed = UInt64[42, 0]
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    ct = @elapsed begin
        compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_warmup(
            rng, pos0, ss, imm_r, margs...)
    end
    println("  Compile: $(round(ct, digits=1))s")
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    wt = @elapsed ((_, _, _, _, st) = compiled_fn(rng2, pos0, ss, imm_r, margs...))
    println("  Warmup run: $(round(wt, digits=1))s")
    save_adapted_state(STATE_PATH, "ide_radar_g$(IDE_GRID)", extract_adapted_state(st);
        num_warmup=NUM_WARMUP, produced_by_arm=ARM, warmup_seed=seed,
        initial_step_size=STEP_SIZE,
        extra=Dict{String,Any}("grid" => IDE_GRID, "NG" => NG, "r" => R, "steps" => NT,
                               "max_tree_depth" => MAX_TREE_DEPTH, "data" => IDE_DATA_PATH))

elseif MODE == "ir"
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = with_arm(() -> @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, margs...))
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    save_hlo(joinpath(OUT_DIR, "hlo", "ide_radar_g$(IDE_GRID)_arm$(ARM).mlir"), hlo_str)
end
