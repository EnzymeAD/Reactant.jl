# Multi-Task Bayesian Regression with Shared Parameter-Dependent Covariance
#
# Model: K tasks sharing a covariance that depends on an estimated parameter
#   tau ~ HalfNormal(1)               (shared scale, part of position)
#   Sigma = tau^2 * Sigma_base + I    (depends on position!)
#   beta_k ~ MVN(0, Sigma)            for k = 1..K
#   y_k ~ Normal(X * beta_k, sigma_k) for k = 1..K
#   sigma_k ~ HalfNormal(1)
#
# Each MvNormal logpdf independently computes chol(tau^2 * Sigma_base + I).
# This Cholesky is ACTIVE (depends on tau, which is in the position vector).
# Enzyme's activity analysis cannot eliminate it.
#
# With pre-AD CSE (No SICM path): 5 identical chol() calls merge → 1 active adjoint.
# Without pre-AD CSE (No opt path): 5 separate active chol() adjoints.
# Post-AD CSE in Enzyme postpasses cannot merge them because the product-rule
# expansion creates different intermediate structures for each site.

function multi_task_regression(rng, X, Sigma_base, y1, y2, y3, y4, y5)
    d = size(X, 2)
    n = size(X, 1)

    # Shared scale parameter (active — part of position vector)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)

    # Build parameter-dependent covariance: tau^2 * Sigma_base + I
    tau_2d = reshape(tau, 1, 1)
    I_d = Matrix{Float64}(LinearAlgebra.I, d, d)
    Sigma = (tau_2d .* tau_2d) .* Sigma_base .+ I_d

    # Task 1
    _, beta1 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta1)
    _, sigma1 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma1)
    mu1 = reshape(X * reshape(beta1, d, 1), n)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(mu1, sigma1, (n,)); symbol=:y1)

    # Task 2
    _, beta2 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta2)
    _, sigma2 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma2)
    mu2 = reshape(X * reshape(beta2, d, 1), n)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(mu2, sigma2, (n,)); symbol=:y2)

    # Task 3
    _, beta3 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta3)
    _, sigma3 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma3)
    mu3 = reshape(X * reshape(beta3, d, 1), n)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(mu3, sigma3, (n,)); symbol=:y3)

    # Task 4
    _, beta4 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta4)
    _, sigma4 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma4)
    mu4 = reshape(X * reshape(beta4, d, 1), n)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(mu4, sigma4, (n,)); symbol=:y4)

    # Task 5
    _, beta5 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta5)
    _, sigma5 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma5)
    mu5 = reshape(X * reshape(beta5, d, 1), n)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(mu5, sigma5, (n,)); symbol=:y5)

    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    d = Int(attrs["d"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:d]
    Sigma_raw = data["Sigma_prior"]
    Sigma_base = [Float64(Sigma_raw[i][j]) for i in 1:d, j in 1:d]

    y1 = Float64.(data["Y1"])
    y2 = Float64.(data["Y2"])
    y3 = Float64.(data["Y3"])
    y4 = Float64.(data["Y4"])
    y5 = Float64.(data["Y5"])

    X_rarray = Reactant.to_rarray(X)
    Sigma_rarray = Reactant.to_rarray(Sigma_base)
    model_args = (X_rarray, Sigma_rarray, y1, y2, y3, y4, y5)

    selection = ProbProg.select(
        ProbProg.Address(:tau),
        ProbProg.Address(:beta1), ProbProg.Address(:sigma1),
        ProbProg.Address(:beta2), ProbProg.Address(:sigma2),
        ProbProg.Address(:beta3), ProbProg.Address(:sigma3),
        ProbProg.Address(:beta4), ProbProg.Address(:sigma4),
        ProbProg.Address(:beta5), ProbProg.Address(:sigma5),
    )

    # tau + 5 * (d + 1) parameters total
    position_size = 1 + 5 * (d + 1)

    return (
        model_fn = multi_task_regression,
        model_args = model_args,
        selection = selection,
        position_size = position_size,
        model_name = "Multi-Task Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    d = Int(attrs["d"])

    if init_params !== nothing
        init_tau = Float64.(init_params["tau"])
        init_beta1 = Float64.(init_params["beta1"])
        init_beta2 = Float64.(init_params["beta2"])
        init_beta3 = Float64.(init_params["beta3"])
        init_beta4 = Float64.(init_params["beta4"])
        init_beta5 = Float64.(init_params["beta5"])
        init_sigma1 = Float64.(init_params["sigma1"])
        init_sigma2 = Float64.(init_params["sigma2"])
        init_sigma3 = Float64.(init_params["sigma3"])
        init_sigma4 = Float64.(init_params["sigma4"])
        init_sigma5 = Float64.(init_params["sigma5"])
    else
        init_tau = [1.0]
        init_beta1 = zeros(d)
        init_beta2 = zeros(d)
        init_beta3 = zeros(d)
        init_beta4 = zeros(d)
        init_beta5 = zeros(d)
        init_sigma1 = [1.0]
        init_sigma2 = [1.0]
        init_sigma3 = [1.0]
        init_sigma4 = [1.0]
        init_sigma5 = [1.0]
    end

    return ProbProg.Constraint(
        :tau => init_tau,
        :beta1 => init_beta1, :sigma1 => init_sigma1,
        :beta2 => init_beta2, :sigma2 => init_sigma2,
        :beta3 => init_beta3, :sigma3 => init_sigma3,
        :beta4 => init_beta4, :sigma4 => init_sigma4,
        :beta5 => init_beta5, :sigma5 => init_sigma5,
        :y1 => Float64.(data["Y1"]),
        :y2 => Float64.(data["Y2"]),
        :y3 => Float64.(data["Y3"]),
        :y4 => Float64.(data["Y4"]),
        :y5 => Float64.(data["Y5"]),
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "tau" => collect(vec(trace.choices[:tau])),
        "beta1" => [collect(trace.choices[:beta1][i, :]) for i in 1:size(trace.choices[:beta1], 1)],
        "beta2" => [collect(trace.choices[:beta2][i, :]) for i in 1:size(trace.choices[:beta2], 1)],
        "beta3" => [collect(trace.choices[:beta3][i, :]) for i in 1:size(trace.choices[:beta3], 1)],
        "beta4" => [collect(trace.choices[:beta4][i, :]) for i in 1:size(trace.choices[:beta4], 1)],
        "beta5" => [collect(trace.choices[:beta5][i, :]) for i in 1:size(trace.choices[:beta5], 1)],
        "sigma1" => collect(vec(trace.choices[:sigma1])),
        "sigma2" => collect(vec(trace.choices[:sigma2])),
        "sigma3" => collect(vec(trace.choices[:sigma3])),
        "sigma4" => collect(vec(trace.choices[:sigma4])),
        "sigma5" => collect(vec(trace.choices[:sigma5])),
    )
end
