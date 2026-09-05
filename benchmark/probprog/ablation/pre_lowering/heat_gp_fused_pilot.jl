#!/usr/bin/env julia
#=
FUSED pilot model: heat-equation inverse problem with a GP prior on the initial
condition. NON-TIMED -- this harness produces IR evidence only and deliberately
has NO `bench` mode.

The point of the model is that ONE log density carries all three redundancy
tiers, in this order:

  TIER 0  data-only, O(D^2)     K[i,j] = exp(-|s_i - s_j| / rho)   (Matern-1/2)
                                built from data `s` and fixed `rho`; expected to
                                be hoisted out of every gradient by CSE in all
                                arms.
                                KERNEL HISTORY (2026-08-25). The GATE-3 IR runs
                                recorded in the evidence log used the SQUARED-
                                EXPONENTIAL kernel exp(-0.5*((s_i-s_j)/rho)^2),
                                spelled as motivating/gp_pois_regr.jl spells it.
                                That kernel is NOT positive definite on this grid
                                at any pilot size: LAPACK dpotrf(K) returns
                                info=14 at D=30 and info=10 at D=50 (rho=0.2),
                                min eig -2.8e-16 and -3.5e-15. StablehloToLapack's
                                CholeskyToPotrf keeps only result 0 and never
                                branches on `info`, so the failure is invisible at
                                runtime and every arm silently used a garbage
                                factor. Scaling by tau^2 > 0 cannot fix it (it
                                scales a negative eigenvalue, sign preserved).
                                Matern-1/2 at rho=0.2 is PD with room to spare
                                (D=30: min eig 8.07e-2, cond 1.26e2; D=50: min eig
                                4.90e-2, cond 3.43e2) and is the standard prior for
                                this problem class. GATE 2 reached the same
                                conclusion independently on the Stan side.
                                Only the kernel FORMULA changed: the tier
                                structure, the CholScale target shape
                                `(tau_2d .* tau_2d) .* K_base`, and the absence of
                                jitter are all untouched.
                                NOT YET RE-RUN: the dumps under
                                heatgp_fused_pilot_dumps/ predate this change.
  TIER 2  algebraic, O(D^3)     cov = (tau_2d .* tau_2d) .* K ; C = chol(cov)
                                u0 = C' * u_tilde.  chol(tau^2 K) is the
                                CholeskyScaleFactorizationHLO target: only SICM
                                can split it into sqrt(tau^2) * chol(K) and hoist
                                chol(K) above the sampler. NO jitter (jitter
                                breaks the pure-scale pattern).
  TIER 1  loop-invariant, O(D^3) per step as written
                                M = I + (dt*kappa)*L ; F = chol(M) ; x = F \ x
                                written INSIDE the traced time loop, exactly as
                                heat_inverse.jl writes it. LICM hoists it out of
                                the loop; CholeskyEigenLiftHLO would additionally
                                turn it into a one-off syevd(L).

  position = [z_tau, z_kappa, u_tilde[1..D]]        (length D+2, shape 1 x (D+2))
  tau = exp(z_tau), kappa = exp(z_kappa), all z-space priors standard normal.

Arms (identical knob triple to heat_inverse.jl):
  A  flat expand, LICM off, SICM off
  B  flat expand, standard passes (LICM on), SICM off
  R  region pipeline, SICM at zero iterations   (localiser only)
  C  region pipeline + SICM
  D  flat expand, SICM nominally on but structurally inert (== A)

Usage (from Reactant.jl root):
  JULIA_CI=true CUDA_VISIBLE_DEVICES="" julia --project=benchmark/probprog \
    benchmark/probprog/ablation/pre_lowering/heat_gp_fused_pilot.jl <A|B|R|C|D> <scale> [mode]
  scale: 0 (D=30,T=10,M_obs=5), 1 (D=50,T=20,M_obs=5), 2 (D=50,T=10,M_obs=5)
  mode : ir (default) | point | draws   (draws = run the sampler and print the
         drawn positions, so two arms can be checked for posterior identity; NOT timed)
=#
using Random
using Reactant, LinearAlgebra
using Reactant.ReactantCore: @trace
using Reactant: @allowscalar, @compile
using Reactant.ProbProg
using JSON3

include(joinpath(@__DIR__, "heat_gp_fused_pilot_data.jl"))
include(joinpath(@__DIR__, "prelowering_common.jl"))

# ── CLI ──────────────────────────────────────────────────────────────────────
const ARM = length(ARGS) >= 1 ? uppercase(ARGS[1]) : "B"
const SCALE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
const MODE = length(ARGS) >= 3 ? ARGS[3] : "ir"
@assert ARM in ("A", "B", "R", "C", "D") "arm must be A/B/R/C/D (got $ARM)"
@assert MODE in ("ir", "point", "draws") "mode must be ir|point|draws (got $MODE); this pilot is NON-TIMED"

# arm -> (disable_opt [flat path], disable_sicm, unroll_time_loops, disable_licm)
const ARM_KNOBS = Dict(
    "A" => (true, true, false, true),    # flat expand, LICM off, SICM off
    "B" => (true, true, false, false),   # flat expand, full std passes, SICM off
    "R" => (false, true, false, false),  # region pipeline, SICM at 0 iterations
    "C" => (false, false, false, false), # region pipeline + SICM
    "D" => (true, false, false, true),   # SICM structurally off (== A)
)
const ARM_DISABLE_OPT, DISABLE_SICM, ARM_UNROLL, ARM_DISABLE_LICM = ARM_KNOBS[ARM]

# ── Config / constants (baked for the traced logpdf) ─────────────────────────
const CFG = heat_gp_config(SCALE)
const D = CFG.D
const T = CFG.T
const DT = CFG.dt
const M_OBS = CFG.M_obs
const S = CFG.S
const OBS_STEP = CFG.obs_step
const SENSOR_IDX = CFG.sensor_idx
const SIGMA = CFG.sigma
const LOG2PI = log(2 * pi)
const P = D + 2                       # position length

# ── Model: fused log joint density ───────────────────────────────────────────
function heat_gp_fused_logpdf(position, s, rho, L, y_data)
    pv = reshape(position, P)
    z_tau = @allowscalar pv[1]
    z_kappa = @allowscalar pv[2]
    z_tau_v = pv[1:1]                       # length 1, keeps the 1x1 scale shape
    u_tilde = pv[3:P]                       # length D
    kappa = exp(z_kappa)

    # ── TIER 0: data-only Matern-1/2 kernel, O(D^2) ─────────────────────────
    s_col = reshape(s, D, 1)
    s_row = reshape(s, 1, D)
    delta = s_col .- s_row
    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-1.0 .* (abs.(delta) ./ rho_2d))

    # ── TIER 2: chol of the parameter-scaled kernel, O(D^3) ─────────────────
    tau_2d = reshape(exp.(z_tau_v), 1, 1)
    cov = (tau_2d .* tau_2d) .* K_base      # exactly CholScale's target shape
    C = cholesky(cov)
    u0 = C.factors' * reshape(u_tilde, D, 1)   # D x 1 initial condition

    # ── TIER 1: implicit Euler, chol INSIDE the traced time loop, O(D^3)/step
    I_d = Matrix{Float64}(I, D, D)
    xcol = u0
    loglik = -1.0 * (M_OBS * S) * (0.5 * LOG2PI + log(SIGMA))
    for j in 1:M_OBS
        @trace track_numbers = false for t in 0:(OBS_STEP - 1)
            M = I_d .+ (DT * kappa) .* L    # 1*I + (dt*kappa)*L, EigenLift shape
            F = cholesky(M)                 # loop-invariant O(D^3)
            xcol = F \ xcol                 # 2 triangular solves, D x 1
        end
        obs_j = xcol[SENSOR_IDX, 1]
        resid = y_data[j, :] .- obs_j
        loglik = loglik - 0.5 * sum(resid .* resid) / (SIGMA * SIGMA)
    end

    logprior = -0.5 * z_tau * z_tau - 0.5 * z_kappa * z_kappa - LOG2PI
    logprior_u = -0.5 * sum(u_tilde .* u_tilde) - 0.5 * D * LOG2PI
    return logprior + logprior_u + loglik
end

const MAX_TREE_DEPTH = parse(Int, get(ENV, "HEATGP_TREE_DEPTH", "3"))

function run_mcmc(rng, pos, ss, imm, s, rho, L, y_data; num_samples=2)
    ProbProg.mcmc_logpdf(rng, heat_gp_fused_logpdf, pos, s, rho, L, y_data;
        algorithm=:NUTS, step_size=ss, inverse_mass_matrix=imm,
        max_tree_depth=MAX_TREE_DEPTH,
        num_warmup=0, num_samples=num_samples,
        adapt_step_size=false, adapt_mass_matrix=false)
end

# ── Setup ────────────────────────────────────────────────────────────────────
Reactant.set_default_backend("cpu")
const DISABLE_OPT = get(ENV, "HEATGP_DISABLE_OPT", ARM_DISABLE_OPT ? "1" : "0") == "1"
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = DISABLE_OPT
Reactant.Compiler.DEBUG_PROBPROG_DISABLE_SICM[] = DISABLE_SICM
const UNROLL = parse(Int, get(ENV, "HEATGP_UNROLL", ARM_UNROLL ? "200" : "0"))
if UNROLL > 0
    Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = UNROLL
end
const DISABLE_LICM =
    get(ENV, "HEATGP_DISABLE_LICM", ARM_DISABLE_LICM ? "1" : "0") == "1"
const COMPILE_OPTS = Reactant.CompileOptions(;
    optimization_passes=:probprog,
    disable_licm_optimization_passes=DISABLE_LICM,
    sync=true,
)

println("=== Heat+GP fused pilot: arm $ARM (path ", DISABLE_OPT ? "flat" : "region",
        ", licm ", DISABLE_LICM ? "OFF" : "ON",
        ", SICM ", DISABLE_SICM ? "OFF" : "ON",
        ", unroll=", UNROLL, "), scale $SCALE, ",
        "D=$D T=$T M_obs=$M_OBS obs_step=$OBS_STEP S=$S P=$P, ",
        "max_tree_depth=$MAX_TREE_DEPTH, mode=$MODE ===")

data = heat_gp_generate_data(CFG)
data_path = heat_gp_data_path(SCALE)
heat_gp_save_data(data_path, CFG, data)
println("  data: kappa_true=$(CFG.kappa_true) rho=$(CFG.rho) sigma=$(CFG.sigma) dt=$(round(DT, sigdigits=4))")
println("  data file: $data_path")

s_r = Reactant.to_rarray(data.s)                              # length D
rho_r = Reactant.to_rarray([CFG.rho])                          # length 1
L_r = Reactant.to_rarray(heat_laplacian(D))                    # D x D
y_r = Reactant.to_rarray(data.y)                               # M_obs x S
imm_r = Reactant.to_rarray(Matrix{Float64}(I, P, P))           # dense identity
pos0 = Reactant.to_rarray(zeros(1, P))                         # 1 x (D+2)
const STEP_SIZE = parse(Float64, get(ENV, "HEATGP_STEP", "0.02"))
ss = Reactant.ConcreteRNumber(STEP_SIZE)

# ── Modes ────────────────────────────────────────────────────────────────────
if MODE == "point"
    println("  compiling point evaluator (optimize=:probprog)...")
    eval_lp(p, s, rho, L, y) = heat_gp_fused_logpdf(p, s, rho, L, y)
    p_r = Reactant.to_rarray(zeros(1, P))
    compiled_lp = @compile compile_options = COMPILE_OPTS eval_lp(p_r, s_r, rho_r, L_r, y_r)
    for (name, pvec) in (("zeros", zeros(P)),
                         ("tau=+0.3,kappa=-0.2,u=0.1", vcat([0.3, -0.2], fill(0.1, D))),
                         ("tau=-0.5,kappa=+0.1,u=-0.05", vcat([-0.5, 0.1], fill(-0.05, D))))
        p_i = Reactant.to_rarray(reshape(pvec, 1, P))
        val = Float64(compiled_lp(p_i, s_r, rho_r, L_r, y_r))
        println("    logpdf($name) = $val   finite=$(isfinite(val))")
    end

elseif MODE == "draws"
    # Non-timed correctness cross-check: run the sampler and print the draws so
    # two arms can be compared for posterior identity. No timing is taken here
    # and none may be read off this mode.
    seed = UInt64[1, 5]
    num_samples = 5
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    println("  compiling sampler (arm $ARM, num_samples=$num_samples)...")
    compiled_fn = @compile sync = true compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, s_r, rho_r, L_r, y_r; num_samples=num_samples)
    rng2 = ProbProg.ReactantRNG(Reactant.to_rarray(copy(seed)))
    samples, diagnostics, _, _ = compiled_fn(rng2, pos0, ss, imm_r, s_r, rho_r, L_r, y_r)
    sa = Array(samples)
    da = Array(diagnostics)
    println("  draws (z_tau, z_kappa, u_tilde[1], u_tilde[end]):")
    for j in axes(sa, 1)
        println("    [$j] z_tau=$(sa[j, 1])  z_kappa=$(sa[j, 2])  ",
                "u1=$(sa[j, 3])  uD=$(sa[j, P])   accepted=$(da[j])")
    end
    out = Dict("arm" => ARM, "scale" => SCALE, "seed" => Int.(seed),
        "step_size" => STEP_SIZE, "max_tree_depth" => MAX_TREE_DEPTH,
        "samples" => [collect(sa[j, :]) for j in axes(sa, 1)],
        "accepted" => Int.(da))
    op = joinpath(heat_gp_out_dir(), "heatgp_draws_arm$(ARM)_scale$(SCALE).json")
    open(io -> JSON3.write(io, out), op, "w")
    println("  wrote draws: $op   n_accepted=$(sum(da))")

else  # ir
    rng = ProbProg.ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
    hlo = @code_hlo compile_options = COMPILE_OPTS run_mcmc(
        rng, pos0, ss, imm_r, s_r, rho_r, L_r, y_r)
    hlo_str = sprint(show, hlo)
    ir_report(hlo_str)
    hlo_path = joinpath(heat_gp_out_dir(), "heatgp_hlo_arm$(ARM)_scale$(SCALE).mlir")
    open(io -> print(io, hlo_str), hlo_path, "w")
    println("  saved HLO: $hlo_path")
end
