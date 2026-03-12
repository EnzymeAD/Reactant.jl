# Turing.jl model spec for Bayesian Logistic Regression
# Matches NumPyro logistic_regression.py and Impulse standard/logistic_regression.jl

using Turing, Distributions, LinearAlgebra

@model function turing_logistic_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc)
    alpha ~ Normal(0.0, alpha_scale)
    beta ~ filldist(Normal(beta_loc, beta_scale), k)
    mu = alpha .+ X * beta
    Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(mu), Y))
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

    Y = Float64.(data["Y"])

    model = turing_logistic_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc)

    return (
        turing_model = model,
        model_name = "Logistic Regression",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    alpha = Array(chain[:alpha])[start:total]
    k = size(chain[Symbol("beta[1]")], 1) > 0 ? length([s for s in names(chain, :parameters) if startswith(string(s), "beta[")]) : 0
    beta = hcat([Array(chain[Symbol("beta[$i]")])[start:total] for i in 1:k]...)

    return Dict{String,Any}(
        "alpha" => collect(alpha),
        "beta" => [collect(beta[i, :]) for i in axes(beta, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    beta = Float64.(init_params["beta"])
    # Return a NamedTuple matching Turing's variable names
    return (alpha=alpha, beta=beta)
end
