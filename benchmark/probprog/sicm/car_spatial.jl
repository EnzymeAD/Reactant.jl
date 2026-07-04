# Leroux CAR Spatial Model for Disease Mapping (Epidemiology)
#
# Model:
#   y_i ~ Poisson(E_i * exp(alpha + phi_i))
#   phi ~ MVN(0, sigma_phi^2 * Q_inv)
#   Q = rho_fixed * (D - W) + (1 - rho_fixed) * I
#   Q_inv precomputed from adjacency graph (sample-invariant)
#   alpha ~ Normal(0, 10)
#   sigma_phi ~ HalfNormal(1)
#
# SICM: **ScaleFamily cascade** on sigma_phi^2 * Q_inv
#   1. CholeskyScaleFactorization: chol(sigma_phi^2 * Q_inv) -> sigma_phi * chol(Q_inv)
#   2. TriangularSolveScaleFactorization on downstream solves
#   3. LogMultiplyDistribution on log(diag(...))
#   Hoists O(N^3) Cholesky of Q_inv; per-iteration O(N^2) solve (phi is latent).
#
# Domain: Spatial epidemiology, disease mapping.
#   The Leroux CAR model (Leroux, Lei & Breslow 2000) is the standard for
#   areal disease mapping, used worldwide in cancer atlases, COVID-19
#   surveillance, and public health reporting (Moraga 2019).
#
#   W = binary adjacency matrix from administrative geography (counties, health
#   districts, census tracts). This is fixed for a given geographic partition
#   and never changes across MCMC iterations.
#
#   Q = rho*(D-W) + (1-rho)*I is the precision matrix of the Leroux CAR prior.
#   rho controls spatial smoothing:
#     rho = 0: independent random effects (Q = I)
#     rho = 1: intrinsic CAR / ICAR (Q = D - W, improper)
#   We fix rho (e.g., rho = 0.99) following empirical Bayes practice common
#   in routine surveillance (Wakefield 2007, Riebler et al. 2016), making
#   Q_inv sample-invariant and enabling the ScaleFamily SICM cascade.
#
#   The Poisson-lognormal likelihood (y_i ~ Poisson(E_i * exp(alpha + phi_i)))
#   is the canonical disease mapping specification (Besag, York & Mollie 1991).
#   E_i = expected counts from indirect standardization (age/sex adjusted),
#   alpha = overall log-relative risk, phi_i = spatial random effect.
#
# Analogues: Stan's car-iar-poisson, R-INLA BYM/BYM2, CARBayes, nimble dcar.

function model_car_spatial(rng, Q_inv, log_E, y)
    n = Base.length(log_E)

    _, alpha = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:alpha
    )

    _, sigma_phi = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_phi
    )

    # Spatial covariance: sigma_phi^2 * Q_inv — ScaleFamily cascade triggers
    sp_2d = reshape(sigma_phi, 1, 1)
    Sigma_phi = (sp_2d .* sp_2d) .* Q_inv

    _, phi = ProbProg.sample(
        rng, ProbProg.MultiNormal(zeros(n), Sigma_phi, (n,)); symbol=:phi
    )

    # Poisson likelihood: y_i ~ Poisson(E_i * exp(alpha + phi_i))
    log_rate = log_E .+ reshape(alpha, 1) .+ phi
    _, obs = ProbProg.sample(
        rng, ProbProg.Poisson(exp.(log_rate), (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    Q_inv_raw = data["Q_inv"]
    Q_inv = [Float64(Q_inv_raw[i][j]) for i in 1:n, j in 1:n]
    log_E = Float64.(data["log_E"])
    y = Float64.(data["Y"])

    Q_inv_rarray = Reactant.to_rarray(Q_inv)
    log_E_rarray = Reactant.to_rarray(log_E)
    model_args = (Q_inv_rarray, log_E_rarray, y)

    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:sigma_phi),
        ProbProg.Address(:phi),
    )

    return (
        model_fn = model_car_spatial,
        model_args = model_args,
        selection = selection,
        position_size = 2 + n,  # alpha(1) + sigma_phi(1) + phi(n)
        model_name = "CAR Spatial",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_sigma_phi = Float64.(init_params["sigma_phi"])
        init_phi = Float64.(init_params["phi"])
    else
        init_alpha = [0.0]
        init_sigma_phi = [1.0]
        init_phi = zeros(n)
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :sigma_phi => init_sigma_phi,
        :phi => init_phi,
        :y => y,
    )
end

function extract_samples(trace)
    phi_samples = trace.choices[:phi]
    return Dict{String,Any}(
        "alpha" => collect(vec(trace.choices[:alpha])),
        "sigma_phi" => collect(vec(trace.choices[:sigma_phi])),
        "phi" => [collect(phi_samples[i, :]) for i in 1:size(phi_samples, 1)],
    )
end
