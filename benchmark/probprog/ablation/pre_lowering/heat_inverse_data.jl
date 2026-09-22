#!/usr/bin/env julia
#=
Shared configuration + deterministic data generation for the 1D heat-equation
Bayesian inverse benchmark (Phase 1, v1 custom logpdf).

Physics: temperature field on a rod [0,1], D interior grid points,
h = 1/(D+1). L = (1/h^2) * tridiag(-1, 2, -1) is the SPD negative Laplacian
(Dirichlet). Implicit Euler with step dt:

    M(kappa) = I + dt*kappa*L         (SPD, D x D)
    M(kappa) * x_{t+1} = x_t          (one linear solve per step)

Data is generated on a 2x finer grid (2D+1 interior points, dt/2) with the true
kappa, restricted to the coarse sensor locations at the coarse observation
times, plus Gaussian noise (inverse-crime avoidance). The generated y matrix and
the coarse initial condition are written to a JSON file so the NumPyro reference
script consumes bit-identical data (L and the scalar constants are recomputed
identically on both sides from exact integer arithmetic).

No Reactant here: plain Julia + JSON3 only.
=#
using Random, LinearAlgebra, JSON3

# Scale presets (design doc rows). scale 0 is a tiny smoke-test config.
function heat_config(scale::Int)
    if scale == 0
        D, T, M_obs = 20, 10, 5      # smoke test
    elseif scale == 1
        D, T, M_obs = 100, 100, 10   # Row 1: LICM-dominant
    elseif scale == 2
        D, T, M_obs = 300, 30, 6     # EigenLift-dominant regime (SICM tier)
    elseif scale == 3
        D, T, M_obs = 300, 100, 10   # same T as Row 1, 3x D
    elseif scale == 4
        D, T, M_obs = 60, 100, 10    # same T as Row 1, 0.6x D
    elseif scale == 5
        D, T, M_obs = 60, 50, 10     # Cubic-Kernel-(small)-style scale
    elseif scale == 6
        D, T, M_obs = 1000, 100, 10  # size sweep 2026-09-10: same T as Row 1, 10x D
    elseif scale == 7
        D, T, M_obs = 300, 400, 10   # size sweep 2026-09-10: same D as Row 2, 4x T
    else
        error("unknown scale $scale (use 0-7)")
    end
    S = 8
    T_final = 0.25                    # total physical simulation time
    dt = T_final / T
    kappa_true = 0.5
    sigma_true = 0.02
    # S sensors at evenly spaced coarse interior indices in 1..D.
    sensor_idx = [round(Int, k * (D + 1) / (S + 1)) for k in 1:S]
    @assert allunique(sensor_idx) "sensor indices collided; increase D or decrease S"
    @assert T % M_obs == 0 "T ($T) must be divisible by M_obs ($M_obs)"
    obs_step = T ÷ M_obs
    obs_steps = collect(obs_step:obs_step:T)   # coarse step indices, length M_obs
    return (; D, T, M_obs, S, dt, kappa_true, sigma_true,
            sensor_idx, obs_steps, obs_step, T_final)
end

# Discrete negative Laplacian scaled by 1/h^2. c = (D+1)^2 is exact in double
# for the D used here, so this is bit-identical between Julia and Python.
function heat_laplacian(D::Int)
    c = (D + 1.0)^2
    L = zeros(Float64, D, D)
    @inbounds for i in 1:D
        L[i, i] = 2.0 * c
        if i > 1
            L[i, i - 1] = -c
        end
        if i < D
            L[i, i + 1] = -c
        end
    end
    return L
end

# Smooth initial condition: sin bump + localized Gaussian, evaluated at the
# D interior grid points s_i = i*h, i = 1..D.
function heat_initial_condition(D::Int)
    h = 1.0 / (D + 1)
    return [sin(pi * (i * h)) + 0.5 * exp(-100.0 * ((i * h) - 0.3)^2) for i in 1:D]
end

# Generate observations on the 2x finer grid, restrict to coarse sensors at
# coarse observation times, add Normal(0, sigma_true) noise. Deterministic.
function heat_generate_data(cfg; seed::Int=20260709)
    D = cfg.D
    Df = 2 * D + 1                 # fine interior points
    dtf = cfg.dt / 2              # fine time step
    Lf = heat_laplacian(Df)
    xf = heat_initial_condition(Df)
    Mf = Matrix{Float64}(I, Df, Df) .+ (dtf * cfg.kappa_true) .* Lf
    Ff = cholesky(Symmetric(Mf))
    # Coarse observation coarse-step t_j maps to fine step 2*t_j; coarse sensor
    # interior index si maps to fine interior index 2*si (both grids agree on
    # physical position i/(D+1) = 2i/(2D+2)).
    obs_fine_steps = Dict(2 * ts => j for (j, ts) in enumerate(cfg.obs_steps))
    snapshots = Vector{Vector{Float64}}(undef, cfg.M_obs)
    for k in 1:(2 * cfg.T)
        xf = Ff \ xf
        if haskey(obs_fine_steps, k)
            snapshots[obs_fine_steps[k]] = copy(xf)
        end
    end
    rng = MersenneTwister(seed)
    y = zeros(Float64, cfg.M_obs, cfg.S)
    for j in 1:cfg.M_obs
        snap = snapshots[j]
        for (i, si) in enumerate(cfg.sensor_idx)
            y[j, i] = snap[2 * si] + cfg.sigma_true * randn(rng)
        end
    end
    x0_coarse = heat_initial_condition(D)
    return (; y, x0_coarse)
end

function heat_data_path(scale::Int; dir::String=get(ENV, "HEATINV_DATA_DIR",
                                                    "/tmp/sbrantq-heatinv/phase1"))
    mkpath(dir)
    return joinpath(dir, "heat_data_scale$(scale).json")
end

function heat_save_data(path::String, cfg, data)
    obj = Dict(
        "D" => cfg.D, "T" => cfg.T, "M_obs" => cfg.M_obs, "S" => cfg.S,
        "dt" => cfg.dt, "kappa_true" => cfg.kappa_true, "sigma_true" => cfg.sigma_true,
        "sensor_idx" => cfg.sensor_idx, "obs_steps" => cfg.obs_steps,
        # store row-major (Vector of rows) so numpy reconstructs (M_obs, S).
        "y" => [collect(data.y[j, :]) for j in 1:cfg.M_obs],
        "x0_coarse" => data.x0_coarse,
    )
    open(path, "w") do io
        JSON3.write(io, obj)
    end
    return path
end
