# Bayesian Linear Regression with Dense Correlated Prior
#
# Model: beta ~ MVN(0, Sigma_prior)     k-dimensional
#        sigma ~ HalfNormal(1)           scalar
#        y ~ Normal(X * beta, sigma)     n-dimensional
#
# Sigma_prior is a fixed dense k×k positive definite matrix.
# Its Cholesky decomposition is O(k³) and purely data-dependent
# (sample-invariant), so hoist-enzyme-regions can lift it out of
# the AD scope. Without hoisting, Enzyme differentiates through
# the Cholesky every iteration.

function correlated_prior_regression(rng, X, Sigma_prior, y)
    k = size(X, 2)
    n = Base.length(y)

    _, beta = ProbProg.sample(
        rng, ProbProg.MultiNormal(zeros(k), Sigma_prior, (k,)); symbol=:beta
    )

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    # Likelihood: y ~ Normal(X * beta, sigma)
    mu = reshape(X * reshape(beta, k, 1), n)
    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(mu, sigma, (n,)); symbol=:y
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    Sigma_raw = data["Sigma_prior"]
    Sigma_prior = [Float64(Sigma_raw[i][j]) for i in 1:k, j in 1:k]
    y = Float64.(data["Y"])

    X_rarray = Reactant.to_rarray(X)
    Sigma_rarray = Reactant.to_rarray(Sigma_prior)
    model_args = (X_rarray, Sigma_rarray, y)

    selection = ProbProg.select(
        ProbProg.Address(:beta),
        ProbProg.Address(:sigma),
    )

    return (
        model_fn = correlated_prior_regression,
        model_args = model_args,
        selection = selection,
        position_size = k + 1,
        model_name = "Correlated Prior Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_beta = zeros(k)
        init_sigma = [1.0]
    end

    return ProbProg.Constraint(
        :beta => init_beta,
        :sigma => init_sigma,
        :y => y,
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
