#!/usr/bin/env julia
#=
Shared configuration + deterministic data generation for the 1D
advection-diffusion (contaminant transport) Bayesian inverse benchmark.

Physics: solute concentration on a 1D domain [0,1] with D interior grid points,
h = 1/(D+1). Transport combines Fickian diffusion and advection at velocity v:

    dc/dt = -kappa * L c  -  v * G c

L = (1/h^2) * tridiag(-1, 2, -1) is the SPD negative Laplacian (Dirichlet);
G = (1/h) * (I - shift) is the first-order upwind (backward difference)
gradient stencil, correct for v > 0. Time stepping is IMEX: the stiff diffusion
term is taken implicitly and the non-stiff advection term explicitly,

    M(kappa) x_{t+1} = x_t - dt*v*(G x_t),   M(kappa) = I + dt*kappa*L

so each step is one Cholesky of the SPD matrix M plus two triangular solves,
plus one sparse mat-vec for the explicit advection flux. M(kappa) does not
depend on t, so the factorization is loop-invariant while the right-hand side
is not; this is the same LICM pattern the heat rows exercise, in a second
physics with a second inferred transport parameter.

Timestep choice is constrained from both sides and both bounds are reported:
  - dt must EXCEED the explicit-diffusion stability limit h^2/(2 kappa),
    otherwise implicit stepping would not be required and the in-loop
    factorization would be an artificial construction;
  - dt must stay BELOW the advection CFL limit h/v (i.e. CFL = v*dt/h < 1),
    otherwise the explicit half of the IMEX split would be unstable.

Data is generated on a 2x finer grid (2D+1 interior points, dt/2) with the true
parameters, restricted to the coarse sensor locations at coarse observation
times, plus Gaussian noise (inverse-crime avoidance, Kaipio & Somersalo 2005).

No Reactant here: plain Julia + JSON3 only.
=#
using Random, LinearAlgebra, JSON3

# Scale presets. scale 0 is a tiny smoke-test config.
function advdiff_config(scale::Int)
    if scale == 0
        D, T, M_obs = 20, 10, 5        # smoke test
    elseif scale == 1
        D, T, M_obs = 200, 100, 10     # table row
    elseif scale == 2
        D, T, M_obs = 100, 100, 10     # heat-D=100-matched cross-check
    elseif scale == 3
        # Size sweep 2026-09-10. The explicit-upwind advection half of the IMEX
        # scheme is CFL-limited: CFL = v_true * dt * (D + 1) with dt = T_final/T,
        # so raising D at a fixed T raises the CFL number proportionally. At
        # D=500, T=100 the CFL would be 1.879 > 1 and the forward model would be
        # unstable, so T is raised with D to hold the scheme in the SAME regime as
        # the table row (CFL 0.7515 against the row's 0.7538; Peclet 15, T_final,
        # kappa, v and sigma all unchanged).
        D, T, M_obs = 500, 250, 10
    elseif scale == 4
        D, T, M_obs = 1000, 500, 10    # same construction, CFL 0.7508
    else
        error("unknown scale $scale (use 0-4)")
    end
    S = 8
    T_final = 0.25                     # total physical simulation time
    dt = T_final / T
    # Transport regime. These are chosen so the instance is genuinely
    # advection-DOMINATED and both transport parameters are identifiable from
    # the sensor data, while both IMEX stability bounds hold with margin:
    #   Peclet  = v*Ldomain/kappa = 15   (advection dominates diffusion)
    #   CFL     = v*dt/h          = 0.754 < 1   (explicit advection stable)
    #   stiff   = dt/(h^2/2kappa) = 20.2 > 1    (implicit diffusion required)
    # At the earlier kappa=0.5, v=0.5 setting the advective part of the plume
    # displacement was only ~9% of the total and a 10% change in v moved the
    # sensors by 0.04 sigma, i.e. v was effectively unidentified and the model
    # was diffusion in all but name. Here a 10% change in v moves the sensors by
    # 1.07 sigma (chi^2 shift 35.5) and a 10% change in kappa by 0.83 sigma
    # (chi^2 shift 7.1), and 74% of the centre-of-mass displacement is advective.
    kappa_true = 0.10                  # diffusivity
    v_true = 1.50                      # advection velocity (left-to-right)
    sigma_true = 0.02                  # 4% of the peak sensor reading
    sensor_idx = [round(Int, k * (D + 1) / (S + 1)) for k in 1:S]
    @assert allunique(sensor_idx) "sensor indices collided; increase D or decrease S"
    @assert T % M_obs == 0 "T ($T) must be divisible by M_obs ($M_obs)"
    obs_step = T ÷ M_obs
    obs_steps = collect(obs_step:obs_step:T)
    return (; D, T, M_obs, S, dt, kappa_true, v_true, sigma_true,
            sensor_idx, obs_steps, obs_step, T_final)
end

"""Stability diagnostics for a config: explicit-diffusion stiffness ratio and
the explicit-advection CFL number, on both the coarse and the fine (data) grid."""
function advdiff_stability(cfg)
    h = 1.0 / (cfg.D + 1)
    dt_diff_limit = h^2 / (2 * cfg.kappa_true)     # explicit-Euler diffusion limit
    stiffness = cfg.dt / dt_diff_limit             # must be > 1
    cfl = cfg.v_true * cfg.dt / h                  # must be < 1
    hf = 1.0 / (2 * cfg.D + 2)
    dtf = cfg.dt / 2
    stiffness_fine = dtf / (hf^2 / (2 * cfg.kappa_true))
    cfl_fine = cfg.v_true * dtf / hf
    peclet = cfg.v_true * 1.0 / cfg.kappa_true     # domain length is 1
    return (; h, dt_diff_limit, stiffness, cfl, stiffness_fine, cfl_fine, peclet)
end

# Discrete negative Laplacian scaled by 1/h^2. c = (D+1)^2 is exact in double.
function advdiff_laplacian(D::Int)
    c = (D + 1.0)^2
    L = zeros(Float64, D, D)
    @inbounds for i in 1:D
        L[i, i] = 2.0 * c
        i > 1 && (L[i, i - 1] = -c)
        i < D && (L[i, i + 1] = -c)
    end
    return L
end

# First-order upwind gradient for v > 0: (G x)_i = (x_i - x_{i-1}) / h, with the
# Dirichlet inflow value x_0 = 0. b = D+1 = 1/h is exact in double.
function advdiff_upwind(D::Int)
    b = D + 1.0
    G = zeros(Float64, D, D)
    @inbounds for i in 1:D
        G[i, i] = b
        i > 1 && (G[i, i - 1] = -b)
    end
    return G
end

# Smooth initial plume: a localized Gaussian pulse near the inflow edge, so the
# advective transport across the domain is visible at the sensors.
function advdiff_initial_condition(D::Int)
    h = 1.0 / (D + 1)
    return [exp(-200.0 * ((i * h) - 0.25)^2) for i in 1:D]
end

# Generate observations on the 2x finer grid, restrict to coarse sensors at
# coarse observation times, add Normal(0, sigma_true) noise. Deterministic.
function advdiff_generate_data(cfg; seed::Int=20260825)
    D = cfg.D
    Df = 2 * D + 1
    dtf = cfg.dt / 2
    Lf = advdiff_laplacian(Df)
    Gf = advdiff_upwind(Df)
    xf = advdiff_initial_condition(Df)
    Mf = Matrix{Float64}(I, Df, Df) .+ (dtf * cfg.kappa_true) .* Lf
    Ff = cholesky(Symmetric(Mf))
    obs_fine_steps = Dict(2 * ts => j for (j, ts) in enumerate(cfg.obs_steps))
    snapshots = Vector{Vector{Float64}}(undef, cfg.M_obs)
    for k in 1:(2 * cfg.T)
        xf = Ff \ (xf .- (dtf * cfg.v_true) .* (Gf * xf))
        if haskey(obs_fine_steps, k)
            snapshots[obs_fine_steps[k]] = copy(xf)
        end
    end
    rng = MersenneTwister(seed)
    y = zeros(Float64, cfg.M_obs, cfg.S)
    for j in 1:cfg.M_obs
        snap = snapshots[j]
        for (i, si) in enumerate(cfg.sensor_idx)
            # coarse interior index si sits at the same physical position as
            # fine interior index 2*si (i/(D+1) = 2i/(2D+2)).
            y[j, i] = snap[2 * si] + cfg.sigma_true * randn(rng)
        end
    end
    x0_coarse = advdiff_initial_condition(D)
    return (; y, x0_coarse)
end

function advdiff_data_path(scale::Int; dir::String=get(ENV, "ADVDIFF_DATA_DIR",
                                                       "/tmp/sbrantq-advdiff"))
    mkpath(dir)
    return joinpath(dir, "advdiff_data_scale$(scale).json")
end

function advdiff_save_data(path::String, cfg, data)
    st = advdiff_stability(cfg)
    obj = Dict(
        "D" => cfg.D, "T" => cfg.T, "M_obs" => cfg.M_obs, "S" => cfg.S,
        "dt" => cfg.dt, "kappa_true" => cfg.kappa_true, "v_true" => cfg.v_true,
        "sigma_true" => cfg.sigma_true,
        "sensor_idx" => cfg.sensor_idx, "obs_steps" => cfg.obs_steps,
        "h" => st.h, "dt_explicit_diffusion_limit" => st.dt_diff_limit,
        "stiffness_ratio" => st.stiffness, "advection_cfl" => st.cfl,
        "stiffness_ratio_fine" => st.stiffness_fine,
        "advection_cfl_fine" => st.cfl_fine, "peclet" => st.peclet,
        "y" => [collect(data.y[j, :]) for j in 1:cfg.M_obs],
        "x0_coarse" => data.x0_coarse,
    )
    open(path, "w") do io
        JSON3.write(io, obj)
    end
    return path
end
