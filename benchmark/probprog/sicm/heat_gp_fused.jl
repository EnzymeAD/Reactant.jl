# Fused benchmark (benchmark B): 1D heat-equation Bayesian inverse problem with a
# Gaussian-process prior on the initial condition.
#
# Carries all three redundancy tiers in ONE model:
#   TIER 0  data-only, O(D^2)   K[i,j] = exp(-|s_i - s_j|/rho) from s and rho
#   TIER 1  sample-dependent but source-loop-INVARIANT, O(D^3)
#           cholesky(M(kappa)), M = I + dt*kappa*Lap, written INSIDE the traced
#           time loop. Standard LICM (`cholesky_licm`) recovers it pre-lowering;
#           SlicStan's three-level placement structurally cannot.
#   TIER 2  algebraic, O(D^3)   cholesky(tau^2 * K)  -> CholeskyScaleFactorization
#           and M = 1*I + (dt*kappa)*Lap -> CholeskyEigenLift
#
# Kernel is Matern-1/2, NOT squared-exponential: the SE kernel on a D-point unit
# grid is not positive definite at any usable lengthscale for D >= 100 (and fails
# even at D=20 for rho=0.2), and neither StablehloToLapack nor Stan reports a
# failed dpotrf -- it just produces a poisoned gradient. See
# oopsla/evidence/related_work/fused_ladder_pilot_2026-08-25.md.
#
# Observation-WINDOW form: T steps split into M_obs windows of obs_step steps.
# obs_step MUST exceed Reactant's WHILE_UNROLL_THRESHOLD (4,
# src/compiler/OptimizationPasses.jl:35) or the traced loop is unrolled and the
# tier-1 rung collapses. The likelihood sits BETWEEN windows, at trace top level,
# because the region/SICM path does not support sample sites inside a traced loop.

using LinearAlgebra

"""Coerce a JSON-decoded 2-D field (vector of rows, or a matrix) to Matrix{Float64}."""
function _hgf_matrix(v)
    if v isa AbstractMatrix
        return Matrix{Float64}(v)
    end
    rows = [Float64.(collect(r)) for r in v]
    return Matrix{Float64}(reduce(vcat, (permutedims(r) for r in rows)))
end

"""Row-major flatten, so Julia and the Stan/python side agree on ordering."""
_hgf_flat_rows(m::AbstractMatrix) = vec(permutedims(m))

# Loop counts must be plain Julia Ints at trace time. They are module-level and
# set by `setup`, mirroring ablation/pre_lowering/heat_inverse.jl -- capturing
# them in a closure makes @trace try to carry them as loop-carried values
# (`MethodError: no method matching set_mlir_data!(::Int64, ...)`).
const HGF_M_OBS = Ref(0)
const HGF_OBS_STEP = Ref(0)
const HGF_ID = Ref{Matrix{Float64}}()

function model_heat_gp_fused(rng, s, Lap, H, rho, dt_r, sigma_r)
    D = Base.length(s)
    S = size(H, 1)
    M_obs = HGF_M_OBS[]
    obs_step = HGF_OBS_STEP[]
    I_d = HGF_ID[]

    _, z_tau = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:z_tau)
    _, z_kappa = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:z_kappa)
    _, u_tilde = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (D,)); symbol=:u_tilde)

    # ── TIER 0: data-only Matern-1/2 kernel, O(D^2) ──────────────────────────
    s_col = reshape(s, D, 1)
    s_row = reshape(s, 1, D)
    delta = s_col .- s_row
    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-1.0 .* (abs.(delta) ./ rho_2d))

    # ── TIER 2: GP prior on the initial condition, O(D^3) ────────────────────
    # `(tau_2d .* tau_2d) .* K_base` is exactly CholeskyScaleFactorization's
    # target shape. NO jitter -- jitter breaks the pure-scale pattern.
    tau_2d = reshape(exp.(z_tau), 1, 1)
    cov = (tau_2d .* tau_2d) .* K_base
    C = cholesky(cov)
    xcol = C.factors' * reshape(u_tilde, D, 1)     # D x 1

    # ── TIER 1: implicit Euler, factorization INSIDE the traced loop ─────────
    dtk = @allowscalar dt_r[1] * exp(z_kappa[1])   # dt * kappa, loop-invariant
    snaps = Vector{Any}(undef, M_obs)
    for j in 1:M_obs
        @trace track_numbers = false for t in 0:(obs_step - 1)
            # M(kappa) = 1*I + (dt*kappa)*Lap -- EigenLift's s*A + t*I form.
            M = I_d .+ dtk .* Lap        # D x D, SPD, loop-invariant
            F = cholesky(M)               # O(D^3), loop-invariant
            xcol = F \ xcol               # 2 triangular solves, D x 1
        end
        snaps[j] = reshape(H * xcol, S)   # S sensor readings at window j
    end

    # One observation site, outside every loop. Concatenating the per-window
    # S-vectors in window order gives ROW-major (window-major) flattening,
    # which is exactly what `_hgf_flat_rows` produces for the data.
    mu_flat = reduce(vcat, snaps)
    sigma_s = @allowscalar sigma_r[1]
    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(mu_flat, sigma_s, (M_obs * S,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    D = Int(attrs["n"])
    M_obs = Int(attrs["M_obs"])
    obs_step = Int(attrs["obs_step"])
    S = Int(attrs["S"])
    dt = Float64(attrs["dt"])
    sigma = Float64(attrs["sigma"])
    rho = Float64(attrs["rho"])

    s = Float64.(data["S_loc"])
    Lap = _hgf_matrix(data["Lap"])
    H = _hgf_matrix(data["H"])

    HGF_M_OBS[] = M_obs
    HGF_OBS_STEP[] = obs_step
    HGF_ID[] = Matrix{Float64}(I, D, D)

    model_args = (
        Reactant.to_rarray(s),
        Reactant.to_rarray(Lap),
        Reactant.to_rarray(H),
        Reactant.to_rarray([rho]),
        Reactant.to_rarray([dt]),
        Reactant.to_rarray([sigma]),
    )

    selection = ProbProg.select(
        ProbProg.Address(:z_tau), ProbProg.Address(:z_kappa),
        ProbProg.Address(:u_tilde),
    )

    P = 2 + D
    return (
        model_fn = model_heat_gp_fused,
        model_args = model_args,
        selection = selection,
        position_size = P,
        # DIAGONAL inverse mass. A length-`position_size` VECTOR selects the
        # diagonal-mass path in harness.jl (a P x P matrix would select dense).
        # Dense is not estimable here: P = D+2 = 202 at the benchmark size with
        # only 200 warmup draws, and dense adaptation additionally puts an
        # off-model O(D^3) factorization inside the sampler, which would pollute
        # exactly the per-gradient cost this ladder measures. Matches the Stan
        # rungs' `metric: diag_e`.
        inverse_mass_matrix = ones(Float64, P),
        model_name = "Heat inverse + GP prior (fused ladder)",
    )
end

function build_constraint(data, init_params)
    D = Int(data["attrs"]["n"])
    y_flat = _hgf_flat_rows(_hgf_matrix(data["Y"]))

    if init_params !== nothing
        init_z_tau = Float64.(init_params["z_tau"])
        init_z_kappa = Float64.(init_params["z_kappa"])
        init_u_tilde = Float64.(init_params["u_tilde"])
    else
        init_z_tau = [0.0]
        init_z_kappa = [0.0]
        init_u_tilde = zeros(D)
    end

    return ProbProg.Constraint(
        :z_tau => init_z_tau,
        :z_kappa => init_z_kappa,
        :u_tilde => init_u_tilde,
        :y => y_flat,
    )
end

function extract_samples(trace)
    z_tau = vec(trace.choices[:z_tau])
    z_kappa = vec(trace.choices[:z_kappa])
    u_tilde = trace.choices[:u_tilde]
    return Dict{String,Any}(
        "z_tau" => collect(z_tau),
        "z_kappa" => collect(z_kappa),
        "u_tilde" => [collect(u_tilde[i, :]) for i in 1:size(u_tilde, 1)],
    )
end
