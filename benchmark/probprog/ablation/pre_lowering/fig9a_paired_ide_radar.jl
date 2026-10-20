#=
IDE radar row driver for Fig. 9 panel (a), in the style of the fig9a_paired_<row>.jl drivers of
oopsla/evidence/fig9a_fair_arm_2026-10-07/scripts/ (copied from fig9a_paired_cochlear.jl; the
model and data come from ide_radar_model.jl instead of being inlined, the harness is included
from that directory by absolute path).

Harness (FIG9A_HARNESS, default fig9a_paired3.jl): all variants of the row compiled in one
process and timed interleaved per seed; A = old "without" (LICM off before lowering and after
AD), N = fair "without" (DEBUG_PROBPROG_DISABLE_PRELOWER_LICM: LICM off before lowering only),
B = "with" (full standard passes). Flat expansion and SICM off in all three.
FIG9A_HARNESS=fig9a_ir_stages.jl dumps the module after the pre-lowering passes, after lowering
plus AD, and after the post-AD passes instead of timing.

Usage: PRED_OUT=<dir> [PRED_VARIANTS=A,N,B] [IDE_GRID=41|21] \
  julia -t 1 --project=<env> fig9a_paired_ide_radar.jl B
or, under the fig9a lock wrapper, copy this file into the fig9a scripts directory and run
  IDE_GRID=41 <fig9a scripts>/run_fig9a.sh <outdir> fig9a_paired_ide_radar.jl B
ARGS[1] only selects the shared knobs (pass B). The frozen adapted state is
adapted_states/ide_radar_g<IDE_GRID>.json (ide_radar.jl B warmup); without it only the IR
harness runs, from the maximum-likelihood position with step 0.01 and identity mass.
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

# Absolute, so that a copy of this driver in the fig9a scripts directory (run_fig9a.sh runs
# "$S/<driver>") still finds the model, the data and the adapted state here.
const PL = "/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog/ablation/pre_lowering"
const FIG9A_SCRIPTS = "/mnt4/sbrantq/probprog/oopsla/evidence/fig9a_fair_arm_2026-10-07/scripts"
const FIG9A_HARNESS = get(ENV, "FIG9A_HARNESS", "fig9a_paired3.jl")
include(joinpath(PL, "prelowering_common.jl"))
include(joinpath(FIG9A_SCRIPTS, FIG9A_HARNESS))
include(joinpath(PL, "ide_radar_model.jl"))

const ARM = uppercase(ARGS[1])
const ARM_DISABLE_OPT, ARM_DISABLE_SICM, ARM_DISABLE_LICM = PRELOWER_ARM_KNOBS[ARM]
const MAX_TREE_DEPTH = parse(Int, get(ENV, "IDE_MAX_DEPTH", "10"))
const STATE_PATH = get(ENV, "PRELOWER_STATE",
    joinpath(PL, "adapted_states", "ide_radar_g$(IDE_GRID).json"))

function run_mcmc(rng, pos, ss, imm, margs...; num_samples=20)
    return ProbProg.mcmc_logpdf(rng, ide_logpdf, pos, margs...;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH, num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = ARM_DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = ARM_DISABLE_SICM
mkopts(dl) = Reactant.CompileOptions(; optimization_passes=:probprog,
    disable_licm_optimization_passes=dl, sync=true)
margs = map(Reactant.to_rarray, MARGS_HOST)
if isfile(STATE_PATH)
    ST = load_adapted_state(STATE_PATH)
    ss = Reactant.ConcreteRNumber(ST.step)
    imm_r = Reactant.to_rarray(ST.imm)
    pos0 = Reactant.to_rarray(ST.pos)
    println("frozen-adapted state: $STATE_PATH  step=$(ST.step)")
else
    @assert FIG9A_HARNESS == "fig9a_ir_stages.jl" "adapted state $STATE_PATH missing (run ide_radar.jl B warmup)"
    ss = Reactant.ConcreteRNumber(0.01)
    imm_r = Reactant.to_rarray(Matrix{Float64}(I, NP, NP))
    pos0 = Reactant.to_rarray(ide_fit_position())
    println("no adapted state (IR harness only): ML position, step 0.01, identity mass")
end
println("ide_radar arm $ARM grid $(IDE_GRID) NG=$NG r=$R steps=$NT depth=$MAX_TREE_DEPTH")

_seed(s::Int) = UInt64[s, 0]
_seed(s::Tuple) = UInt64[s[1], s[2]]
rng0() = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
compile_with(n, opts) = @compile sync = true compile_options = opts run_mcmc(
    rng0(), pos0, ss, imm_r, margs...; num_samples=n)
call(fn, seed) = fn(ProbProg.ReactantRNG(Reactant.to_rarray(_seed(seed))),
    pos0, ss, imm_r, margs...)
hlo_with(n, opts) = sprint(show, @code_hlo compile_options = opts run_mcmc(
    rng0(), pos0, ss, imm_r, margs...; num_samples=n))

const ROW_TAG = "ide_radar_g$(IDE_GRID)"
pred_paired()
