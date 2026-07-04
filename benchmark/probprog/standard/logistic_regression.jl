function logistic_regression(rng, X, n, k, alpha_scale, beta_scale, beta_loc)
    _, alpha = ProbProg.sample(
        rng, ProbProg.Normal(0.0, alpha_scale, (1,)); symbol=:alpha
    )
    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(beta_loc, beta_scale, (k,)); symbol=:beta
    )
    logits = alpha .+ X * beta
    _, Y = ProbProg.sample(
        rng, ProbProg.Bernoulli(logits, (n,)); symbol=:Y
    )
    return Y
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    alpha_scale = Float64(attrs["alpha_scale"])
    beta_scale = Float64(attrs["beta_scale"])
    beta_loc = Float64(attrs["beta_loc"])

    X_raw = data["X"]
    X = zeros(Float64, n, k)
    for i in 1:n, j in 1:k
        X[i, j] = Float64(X_raw[i][j])
    end

    X_rarray = Reactant.to_rarray(X)
    model_args = (X_rarray, n, k, alpha_scale, beta_scale, beta_loc)

    selection = ProbProg.select(ProbProg.Address(:alpha), ProbProg.Address(:beta))

    return (
        model_fn = logistic_regression,
        model_args = model_args,
        selection = selection,
        position_size = 1 + k,
        model_name = "Logistic Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    Y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_beta = Float64.(init_params["beta"])
    else
        init_alpha = [0.0]
        init_beta = zeros(k)
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :beta => init_beta,
        :Y => Y,
    )
end

function extract_samples(trace)
    alpha_samples = vec(trace.choices[:alpha])
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "alpha" => collect(alpha_samples),
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
    )
end
