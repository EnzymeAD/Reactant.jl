# Stochastic Volatility Model (Financial Econometrics)
#
# Model (non-centered parameterization):
#   mu ~ Normal(0, 5)               (mean log-volatility)
#   sigma_h ~ HalfNormal(1)         (volatility of log-volatility)
#   eta ~ Normal(0, 1)^T            (standardized innovations)
#   h = mu + sigma_h * L_C * eta    (log-volatility, NCP)
#   y_t ~ Normal(0, exp(h_t / 2))   (returns with time-varying volatility)
#
#   C_ar1 = AR(1) stationary covariance with fixed persistence phi:
#     C_ar1[s,t] = phi^|s-t| / (1 - phi^2)
#   C_ar1 is precomputed and sample-invariant.
#
# SICM: **CholeskyScaleFactorization** on sigma_h^2 * C_ar1
#   chol(sigma_h^2 * C_ar1) -> sigma_h * chol(C_ar1)
#   Hoists O(T^3) Cholesky of Toeplitz C_ar1; per-iteration O(T^2) matvec.
#
#   Downstream cascade:
#     TriangularSolveScaleFactorization on tri_solve(sigma_h * L, ...)
#     LogMultiplyDistribution on log(diag(sigma_h * L))
#
# Domain: Financial econometrics, volatility modeling.
#   Canonical stochastic volatility model following Kim, Shephard & Chib (1998).
#   The AR(1) structure for log-volatility is the standard specification in
#   financial time series (Taylor 1986, Harvey, Ruiz & Shephard 1994).
#
#   phi (persistence) is fixed via empirical Bayes: estimated from the sample
#   autocorrelation of squared returns, then conditioned upon for the full
#   Bayesian inference. This two-stage approach is standard in applied work
#   (e.g., Jacquier, Polson & Rossi 2004) and in benchmark implementations
#   (Stan manual Section 2.2). Fixing phi makes C_ar1 sample-invariant,
#   enabling the CholeskyScaleFactorization SICM pattern. Typical values:
#   phi in [0.9, 0.99] for daily financial returns.
#
# Structural differences from GP models:
#   - Toeplitz covariance from AR(1) temporal structure (not a kernel)
#   - Heteroscedastic observation: y_t ~ Normal(0, exp(h_t/2))
#   - Time series context (ordered observations, not exchangeable)

function model_stochastic_volatility(rng, C_ar1, y)
    T = Base.length(y)

    _, mu = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 5.0, (1,)); symbol=:mu
    )

    _, sigma_h = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_h
    )

    _, eta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (T,)); symbol=:eta
    )

    # NCP: h = mu + sigma_h * L_C * eta
    # Covariance: sigma_h^2 * C_ar1 -> CholeskyScaleFactorization
    sh_2d = reshape(sigma_h, 1, 1)
    cov_h = (sh_2d .* sh_2d) .* C_ar1

    C = cholesky(cov_h)
    h = reshape(mu, 1) .+ reshape(C.factors' * reshape(eta, T, 1), T)

    # Observation: y_t ~ Normal(0, exp(h_t / 2))
    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(0.0, exp.(h ./ 2.0), (T,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    T = Int(attrs["T"])
    phi = Float64(attrs["phi"])

    # Precompute AR(1) stationary covariance: C[s,t] = phi^|s-t| / (1 - phi^2)
    C_ar1_raw = data["C_ar1"]
    C_ar1 = [Float64(C_ar1_raw[i][j]) for i in 1:T, j in 1:T]
    y = Float64.(data["Y"])

    C_ar1_rarray = Reactant.to_rarray(C_ar1)
    model_args = (C_ar1_rarray, y)

    selection = ProbProg.select(
        ProbProg.Address(:mu),
        ProbProg.Address(:sigma_h),
        ProbProg.Address(:eta),
    )

    return (
        model_fn = model_stochastic_volatility,
        model_args = model_args,
        selection = selection,
        position_size = 2 + T,  # mu(1) + sigma_h(1) + eta(T)
        model_name = "Stochastic Volatility",
    )
end

function build_constraint(data, init_params)
    T = Int(data["attrs"]["T"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_mu = Float64.(init_params["mu"])
        init_sigma_h = Float64.(init_params["sigma_h"])
        init_eta = Float64.(init_params["eta"])
    else
        init_mu = [0.0]
        init_sigma_h = [0.5]
        init_eta = zeros(T)
    end

    return ProbProg.Constraint(
        :mu => init_mu,
        :sigma_h => init_sigma_h,
        :eta => init_eta,
        :y => y,
    )
end

function extract_samples(trace)
    eta_samples = trace.choices[:eta]
    return Dict{String,Any}(
        "mu" => collect(vec(trace.choices[:mu])),
        "sigma_h" => collect(vec(trace.choices[:sigma_h])),
        "eta" => [collect(eta_samples[i, :]) for i in 1:size(eta_samples, 1)],
    )
end
