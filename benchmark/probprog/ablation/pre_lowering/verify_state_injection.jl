#!/usr/bin/env julia
#=
State-injection verification for the frozen-adapted-state protocol.

Runs the full checklist on ONE cheap model at smoke scale (heat scale 0,
D=20, T=10, 2 parameters) before any measured batch is allowed to run:

  1. Injection takes effect, not merely accepted. Positive controls: perturb
     the position, the step size and the inverse mass matrix independently and
     confirm the trajectory changes; and confirm the injected inverse mass is
     NOT silently replaced by the identity default.
  2. Shape / convention round-trip. Under the rank-determines-diagonal
     convention a dense diag(a,b) and a rank-1 [a,b] describe the SAME metric,
     so they must produce identical trajectories; and a full dense matrix must
     differ from its own diagonal, which is what catches a silent flattening.
     A transpose of the adapted matrix is unobservable because the adapted
     matrix is a symmetric covariance estimate; that is stated, not tested.
  3. Arms-equivalence. With the same saved state and matched seeds the two
     arms must make bit-identical acceptance decisions and ULP-close samples.
     This is the gate the protocol exists to restore.
  4. Continuation equivalence (statistical, not bit-identical). A
     warmup-then-restore run must sit in the same regime as the post-warmup
     phase of a single continuous adaptation-on run: comparable acceptance
     rate, comparable per-sample cost (the observable proxy for tree depth,
     which the diagnostics array does not expose), consistent posterior.

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/verify_state_injection.jl [scale]
=#
using Random
using Reactant, LinearAlgebra, Statistics
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "heat_inverse_data.jl"))
include(joinpath(@__DIR__, "prelowering_common.jl"))

const SCALE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
const NW = parse(Int, get(ENV, "PRELOWER_WARMUP", "50"))
const NS = 20

const CFG = heat_config(SCALE)
const D = CFG.D
const T = CFG.T
const DT = CFG.dt
const M_OBS = CFG.M_obs
const S = CFG.S
const OBS_STEP = CFG.obs_step
const SENSOR_IDX = CFG.sensor_idx
const LOG2PI = log(2 * pi)

function heat_inverse_logpdf(position, L, x0col, y_data)
    z_kappa = @allowscalar position[1]
    z_sigma = @allowscalar position[2]
    kappa = exp(z_kappa)
    sigma = exp(z_sigma)
    I_d = Matrix{Float64}(I, D, D)
    xcol = x0col
    loglik = -1.0 * (M_OBS * S) * (0.5 * LOG2PI + z_sigma)
    for j in 1:M_OBS
        @trace track_numbers = false for t in 0:(OBS_STEP - 1)
            M = I_d .+ (DT * kappa) .* L
            F = cholesky(M)
            xcol = F \ xcol
        end
        obs_j = xcol[SENSOR_IDX, 1]
        resid = y_data[j, :] .- obs_j
        loglik = loglik - 0.5 * sum(resid .* resid) / (sigma * sigma)
    end
    logprior = -0.5 * z_kappa * z_kappa - 0.5 * z_sigma * z_sigma - LOG2PI
    return logprior + loglik
end

measure(rng, pos, ss, imm, L, x0, y) = ProbProg.mcmc_logpdf(
    rng, heat_inverse_logpdf, pos, L, x0, y;
    algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
    num_warmup=0, num_samples=NS, adapt_step_size=false, adapt_mass_matrix=false)

warmup_only(rng, pos, ss, imm, L, x0, y) = ProbProg.mcmc_logpdf(
    rng, heat_inverse_logpdf, pos, L, x0, y;
    algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
    num_warmup=NW, num_samples=1, adapt_step_size=true, adapt_mass_matrix=true)

continuous(rng, pos, ss, imm, L, x0, y) = ProbProg.mcmc_logpdf(
    rng, heat_inverse_logpdf, pos, L, x0, y;
    algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm, max_tree_depth=5,
    num_warmup=NW, num_samples=NS, adapt_step_size=true, adapt_mass_matrix=true)

Reactant.set_default_backend("cpu")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = true

opts(disable_licm) = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=disable_licm, sync=true)

cfg_data = heat_generate_data(CFG)
L_r = Reactant.to_rarray(heat_laplacian(D))
x0_r = Reactant.to_rarray(reshape(cfg_data.x0_coarse, D, 1))
y_r = Reactant.to_rarray(cfg_data.y)
P = 2                                   # z_kappa, z_sigma

println("=== State-injection verification: heat scale $SCALE (D=$D, T=$T, P=$P), ",
        "warmup=$NW, num_samples=$NS ===")

seedv() = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[7, 11]))
results = Dict{String,Any}()

# ── Phase 0: produce the adapted state (Pre-Opt arm, adaptation ON) ──────────
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
pos_init = Reactant.to_rarray(reshape(zeros(P), 1, P))
imm_init = Reactant.to_rarray(Matrix{Float64}(I, P, P))
ss_init = Reactant.ConcreteRNumber(0.02)

println("\n[0] warmup with adaptation ON (arm B)")
warm_fn = @compile sync = true compile_options = opts(false) warmup_only(
    seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)
_, _, _, _, st = warm_fn(seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)
AD = extract_adapted_state(st)
println("    adapted step   = $(AD.step)")
println("    adapted invmass= $(AD.imm)   (rank $(ndims(AD.imm)))")
println("    adapted pos    = $(vec(AD.pos))")
results["adapted"] = Dict("step" => AD.step, "imm" => [collect(AD.imm[i, :]) for i in 1:P],
                          "pos" => collect(vec(AD.pos)))

# Compiled measured thunk (dense inverse mass). Values are runtime arguments, so
# ONE compile serves every value perturbation below.
bench_B = @compile sync = true compile_options = opts(false) measure(
    seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)

run_dense(pos, step, imm) = begin
    s, d, ld, _, _ = bench_B(seedv(), Reactant.to_rarray(pos),
                             Reactant.ConcreteRNumber(step),
                             Reactant.to_rarray(imm), L_r, x0_r, y_r)
    (samples=Array(s), accepted=Int.(Array(d)[:, 1]), diverging=Int.(Array(d)[:, 2]),
     logdens=Array(ld))
end

maxdiff(a, b) = maximum(abs.(a .- b))

base = run_dense(AD.pos, AD.step, AD.imm)
println("\n[1] injection positive controls (all must CHANGE the trajectory)")
ctrl = Dict{String,Any}()
for (name, pos, step, imm) in (
        ("position +0.05",  AD.pos .+ 0.05,             AD.step,       AD.imm),
        ("step_size x1.5",  AD.pos,                     AD.step * 1.5, AD.imm),
        ("invmass x2",      AD.pos,                     AD.step,       AD.imm .* 2),
        ("invmass=identity", AD.pos,                    AD.step,       Matrix{Float64}(I, P, P)),
        ("invmass=diag(adapted)", AD.pos,               AD.step,       Matrix(Diagonal(diag(AD.imm)))))
    r = run_dense(pos, step, imm)
    d = maxdiff(r.samples, base.samples)
    same_acc = r.accepted == base.accepted
    ok = d > 0
    ctrl[name] = Dict("max_sample_diff" => d, "accepts_identical_to_base" => same_acc,
                      "changed" => ok)
    println("    $(rpad(name, 24)) max|dsample| = $(d)   changed=$(ok)")
end
results["positive_controls"] = ctrl

# Confirm the harness's own loader reproduces the in-memory state exactly.
tmp_state = joinpath(@__DIR__, "adapted_states", "_verify_scale$(SCALE).json")
save_adapted_state(tmp_state, "verify_heat_scale$(SCALE)", AD; num_warmup=NW,
                   produced_by_arm="B", warmup_seed=UInt64[7, 11],
                   initial_step_size=0.02)
LD = load_adapted_state(tmp_state)
rt = Dict("step_exact" => LD.step == AD.step,
          "imm_exact" => LD.imm == AD.imm,
          "pos_exact" => vec(LD.pos) == vec(AD.pos),
          "imm_rank" => ndims(LD.imm))
println("\n[2a] JSON round-trip exactness: $(rt)")
results["json_roundtrip"] = rt

# Rank convention: dense diag(a,b) and rank-1 [a,b] are the SAME metric.
a, b = 0.013, 0.37
dense_diag = Matrix(Diagonal([a, b]))
r_dense = run_dense(AD.pos, AD.step, dense_diag)
bench_B_diag = @compile sync = true compile_options = opts(false) measure(
    seedv(), pos_init, ss_init, Reactant.to_rarray([a, b]), L_r, x0_r, y_r)
sd, dd, ldd, _, _ = bench_B_diag(seedv(), Reactant.to_rarray(AD.pos),
                                 Reactant.ConcreteRNumber(AD.step),
                                 Reactant.to_rarray([a, b]), L_r, x0_r, y_r)
r_rank1 = (samples=Array(sd), accepted=Int.(Array(dd)[:, 1]))
conv = Dict("dense_vs_rank1_max_sample_diff" => maxdiff(r_dense.samples, r_rank1.samples),
            "dense_vs_rank1_accepts_identical" => r_dense.accepted == r_rank1.accepted,
            # A silent flattening of the dense matrix to its diagonal would make
            # these two agree; they must NOT.
            "full_vs_its_diagonal_max_sample_diff" =>
                ctrl["invmass=diag(adapted)"]["max_sample_diff"],
            "adapted_offdiag_max_abs" => maximum(abs.(AD.imm - Diagonal(diag(AD.imm)))),
            "note_transpose" => "the adapted inverse mass is a symmetric covariance " *
                                "estimate, so a transpose is unobservable and harmless; " *
                                "not tested")
println("\n[2b] rank convention: dense diag(a,b) vs rank-1 [a,b]")
println("    max|dsample| = $(conv["dense_vs_rank1_max_sample_diff"])  ",
        "accepts identical = $(conv["dense_vs_rank1_accepts_identical"])")
results["rank_convention"] = conv

# ── [3] arms-equivalence from the same injected state ────────────────────────
println("\n[3] arms-equivalence from the SAME injected state (the gate)")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
bench_A = @compile sync = true compile_options = opts(true) measure(
    seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)
sA, dA, ldA, _, _ = bench_A(seedv(), Reactant.to_rarray(AD.pos),
                            Reactant.ConcreteRNumber(AD.step),
                            Reactant.to_rarray(AD.imm), L_r, x0_r, y_r)
armA = (samples=Array(sA), accepted=Int.(Array(dA)[:, 1]), diverging=Int.(Array(dA)[:, 2]),
        logdens=Array(ldA))
arms = Dict("accepts_identical" => armA.accepted == base.accepted,
            "diverging_identical" => armA.diverging == base.diverging,
            "max_sample_diff" => maxdiff(armA.samples, base.samples),
            "max_logdens_diff" => maxdiff(armA.logdens, base.logdens),
            "n_accepted" => sum(base.accepted))
println("    accepts identical = $(arms["accepts_identical"])  ",
        "diverging identical = $(arms["diverging_identical"])")
println("    max|dsample| = $(arms["max_sample_diff"])  ",
        "max|dlogdens| = $(arms["max_logdens_diff"])")
results["arms_equivalence"] = arms

# ── [4] continuation equivalence vs one continuous adaptation-on run ─────────
println("\n[4] continuation equivalence (regime, not bit-identity)")
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = true
cont_fn = @compile sync = true compile_options = opts(false) continuous(
    seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)
cont_fn(seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)   # warm the thunk
tc = @elapsed ((sc, dc, ldc, _, stc) = cont_fn(seedv(), pos_init, ss_init, imm_init,
                                               L_r, x0_r, y_r))
cont = (samples=Array(sc), accepted=Int.(Array(dc)[:, 1]), logdens=Array(ldc))
# The continuous run's wall time is dominated by its NW warmup iterations, which
# run at the tiny INITIAL step size and therefore build far deeper trees than any
# post-adaptation iteration. Subtract a (NW warmup + 1 sample) run to isolate the
# per-sample cost of the continuous run's POST-warmup phase, which is the regime
# the restored run is supposed to reproduce.
warm_fn(seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)   # warm the thunk
tw = @elapsed warm_fn(seedv(), pos_init, ss_init, imm_init, L_r, x0_r, y_r)
cont_post_per_sample = 1000 * (tc - tw) / (NS - 1)
run_dense(AD.pos, AD.step, AD.imm)                              # warm the thunk
t_restored = @elapsed run_dense(AD.pos, AD.step, AD.imm)
contstep = Array(stc.step_size)[]
cmp = Dict(
    "continuous_accept_rate" => mean(cont.accepted),
    "restored_accept_rate" => mean(base.accepted),
    "continuous_step_size" => contstep,
    "restored_step_size" => AD.step,
    "continuous_total_ms_incl_warmup" => 1000 * tc,
    "continuous_warmup_only_ms" => 1000 * tw,
    "continuous_POSTwarmup_per_sample_ms" => cont_post_per_sample,
    "restored_per_sample_ms" => 1000 * t_restored / NS,
    "continuous_logdens_mean" => mean(cont.logdens),
    "restored_logdens_mean" => mean(base.logdens),
    "continuous_pos_mean" => [mean(cont.samples[:, i]) for i in 1:P],
    "restored_pos_mean" => [mean(base.samples[:, i]) for i in 1:P],
    "continuous_pos_sd" => [std(cont.samples[:, i]) for i in 1:P],
    "restored_pos_sd" => [std(base.samples[:, i]) for i in 1:P],
)
for k in sort(collect(keys(cmp)))
    println("    $(rpad(k, 28)) $(cmp[k])")
end
results["continuation_equivalence"] = cmp

out = joinpath(@__DIR__, "logs", "state_injection_verification_scale$(SCALE).json")
mkpath(dirname(out))
open(io -> JSON3.write(io, results), out, "w")
println("\nwrote $out")
