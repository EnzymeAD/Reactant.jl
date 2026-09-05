#!/usr/bin/env julia
#=
Config presets + deterministic data generation for the FUSED pilot model
(heat-equation inverse problem with a GP prior on the initial condition).

NON-TIMED PILOT. This file exists to produce IR evidence for the three
redundancy tiers; it does not feed any timing table and it does not touch the
existing heat / advdiff / star harnesses.

Physics is identical to heat_inverse_data.jl (which is `include`d here, never
edited): rod [0,1], D interior grid points at s_i = i/(D+1), h = 1/(D+1),
L = (1/h^2) tridiag(-1,2,-1) SPD, implicit Euler M(kappa) = I + dt*kappa*L.

What is NEW is the prior on the initial condition. Instead of a fixed x0, the
initial temperature field is a non-centred GP draw

    K[i,j] = exp(-|s_i - s_j| / rho)                  (tier 0, data-only, Matern-1/2)
    cov    = tau^2 * K,   C = chol(cov),  u0 = C' * u_tilde   (tier 2)

with u_tilde ~ N(0, I_D) and tau = exp(z_tau). The parameter vector is

    position = [z_tau, z_kappa, u_tilde[1..D]]        (length D+2)

Observations y (M_obs x S) are produced by forward-simulating the SAME coarse
implicit-Euler dynamics from heat_initial_condition(D) with kappa_true, then
adding Normal(0, sigma) noise. Physical realism is not the point of this pilot;
the IR structure is, so no finer-grid inverse-crime avoidance is done here.
=#
using Random, LinearAlgebra, JSON3

include(joinpath(@__DIR__, "heat_inverse_data.jl"))   # heat_laplacian / heat_initial_condition

"""Grid locations s_i = i/(D+1), i = 1..D -- the grid heat_initial_condition uses."""
heat_gp_grid(D::Int) = [i / (D + 1.0) for i in 1:D]

"""Pilot scale presets. Deliberately small: this is an IR-structure pilot."""
function heat_gp_config(scale::Int)
    # obs_step MUST exceed Reactant's WHILE_UNROLL_THRESHOLD (default 4,
    # OptimizationPasses.jl:35) or the traced time loop is unrolled away and the
    # tier-1 loop-invariance question becomes vacuous (arms A and B then produce
    # byte-different but op-count-identical IR). obs_step = 5 everywhere here.
    if scale == 0
        D, T, M_obs = 30, 10, 2      # obs_step 5
    elseif scale == 1
        D, T, M_obs = 50, 20, 4      # obs_step 5
    elseif scale == 2
        D, T, M_obs = 50, 10, 2      # obs_step 5
    else
        error("unknown pilot scale $scale (use 0-2)")
    end
    S = 8
    T_final = 0.25
    dt = T_final / T
    kappa_true = 0.5
    tau_true = 1.0
    rho = 0.2                     # FIXED Matern-1/2 lengthscale (data, not a parameter)
    sigma = 0.02                  # FIXED observation noise (data)
    sensor_idx = [round(Int, k * (D + 1) / (S + 1)) for k in 1:S]
    @assert allunique(sensor_idx) "sensor indices collided"
    @assert T % M_obs == 0 "T ($T) must be divisible by M_obs ($M_obs)"
    obs_step = T ÷ M_obs
    obs_steps = collect(obs_step:obs_step:T)
    return (; D, T, M_obs, S, dt, kappa_true, tau_true, rho, sigma,
            sensor_idx, obs_steps, obs_step, T_final)
end

"""Matern-1/2 (exponential) kernel on the grid, plain Julia (reference only).

Squared-exponential is NOT usable here: on the grid s_i = i/(D+1) it is not
positive definite at any pilot size (dpotrf info=14 at D=30, info=10 at D=50 for
rho=0.2), and the compiled IR discards dpotrf's info result, so the failure is
silent. Matern-1/2 at rho=0.2 has min eig 8.07e-2 (D=30) / 4.90e-2 (D=50)."""
function heat_gp_kernel(D::Int, rho::Float64)
    s = heat_gp_grid(D)
    return [exp(-abs(s[i] - s[j]) / rho) for i in 1:D, j in 1:D]
end

"""Forward-simulate coarse implicit Euler from x0 and read the S sensors at the
M_obs observation times; add Normal(0, sigma) noise."""
function heat_gp_generate_data(cfg; seed::Int=20260825)
    D = cfg.D
    L = heat_laplacian(D)
    M = Matrix{Float64}(I, D, D) .+ (cfg.dt * cfg.kappa_true) .* L
    F = cholesky(Symmetric(M))
    x = heat_initial_condition(D)                 # the "true" u0 for data gen
    obs_at = Dict(ts => j for (j, ts) in enumerate(cfg.obs_steps))
    snapshots = Vector{Vector{Float64}}(undef, cfg.M_obs)
    for k in 1:cfg.T
        x = F \ x
        if haskey(obs_at, k)
            snapshots[obs_at[k]] = copy(x)
        end
    end
    rng = MersenneTwister(seed)
    y = zeros(Float64, cfg.M_obs, cfg.S)
    for j in 1:cfg.M_obs, (i, si) in enumerate(cfg.sensor_idx)
        y[j, i] = snapshots[j][si] + cfg.sigma * randn(rng)
    end
    return (; y, x0_true=heat_initial_condition(D), s=heat_gp_grid(D))
end

function heat_gp_out_dir()
    dir = get(ENV, "HEATGP_OUT_DIR", "/tmp/sbrantq-heatgpfused/pilot")
    mkpath(dir)
    return dir
end

heat_gp_data_path(scale::Int) =
    joinpath(heat_gp_out_dir(), "heat_gp_fused_data_scale$(scale).json")

function heat_gp_save_data(path::String, cfg, data)
    obj = Dict(
        "D" => cfg.D, "T" => cfg.T, "M_obs" => cfg.M_obs, "S" => cfg.S,
        "dt" => cfg.dt, "kappa_true" => cfg.kappa_true, "tau_true" => cfg.tau_true,
        "rho" => cfg.rho, "sigma" => cfg.sigma,
        "sensor_idx" => cfg.sensor_idx, "obs_steps" => cfg.obs_steps,
        "s" => data.s,
        "y" => [collect(data.y[j, :]) for j in 1:cfg.M_obs],
        "x0_true" => data.x0_true,
    )
    open(io -> JSON3.write(io, obj), path, "w")
    return path
end
