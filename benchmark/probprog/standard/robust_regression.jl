function robust_regression(rng, X, Y, n, k, alpha_scale, beta_scale, beta_loc, sigma_mean)
    _, alpha = ProbProg.sample(
        rng, ProbProg.Normal(0.0, alpha_scale, (1,)); symbol=:alpha
    )
    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(beta_loc, beta_scale, (k,)); symbol=:beta
    )
    _, nu = ProbProg.sample(
        rng, ProbProg.Gamma(2.0, 0.1, (1,)); symbol=:nu
    )
    _, sigma = ProbProg.sample(
        rng, ProbProg.Exponential(1.0 / sigma_mean, (1,)); symbol=:sigma
    )
    mu = alpha .+ X * beta
    _, _ = ProbProg.sample(
        rng, ProbProg.StudentT(nu, mu, sigma, (n,)); symbol=:Y
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    alpha_scale = Float64(attrs["alpha_scale"])
    beta_scale = Float64(attrs["beta_scale"])
    beta_loc = Float64(attrs["beta_loc"])
    sigma_mean = Float64(attrs["sigma_mean"])

    X_raw = data["X"]
    X = zeros(Float64, n, k)
    for i in 1:n, j in 1:k
        X[i, j] = Float64(X_raw[i][j])
    end
    Y_raw = Float64.(data["Y"])

    X_rarray = Reactant.to_rarray(X)
    Y_rarray = Reactant.to_rarray(Y_raw)
    model_args = (X_rarray, Y_rarray, n, k, alpha_scale, beta_scale, beta_loc, sigma_mean)

    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:beta),
        ProbProg.Address(:nu),
        ProbProg.Address(:sigma),
    )

    # alpha(1) + beta(k) + nu(1) + sigma(1) = k + 3
    return (
        model_fn = robust_regression,
        model_args = model_args,
        selection = selection,
        position_size = k + 3,
        model_name = "Robust Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    Y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_beta = Float64.(init_params["beta"])
        init_nu = Float64.(init_params["nu"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_alpha = [0.0]
        init_beta = zeros(k)
        init_nu = [20.0]     # Gamma(2, 0.1) mean = 20
        init_sigma = [10.0]  # Exponential(1/sigma_mean) mean = sigma_mean
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :beta => init_beta,
        :nu => init_nu,
        :sigma => init_sigma,
        :Y => Y,
    )
end

function extract_samples(trace)
    alpha_samples = vec(trace.choices[:alpha])
    beta_samples = trace.choices[:beta]
    nu_samples = vec(trace.choices[:nu])
    sigma_samples = vec(trace.choices[:sigma])
    return Dict{String,Any}(
        "alpha" => collect(alpha_samples),
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "nu" => collect(nu_samples),
        "sigma" => collect(sigma_samples),
    )
end
