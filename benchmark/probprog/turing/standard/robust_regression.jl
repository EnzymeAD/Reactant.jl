# Turing.jl model spec for Robust Regression
# Matches NumPyro robust_regression.py and Impulse standard/robust_regression.jl

using Turing, Distributions, LinearAlgebra

@model function turing_robust_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc, sigma_mean)
    alpha ~ Normal(0.0, alpha_scale)
    beta ~ filldist(Normal(beta_loc, beta_scale), k)
    nu ~ Gamma(2.0, 1.0 / 0.1)  # Distributions.jl: Gamma(shape, scale); NumPyro: Gamma(conc, rate) → scale = 1/rate
    sigma ~ Exponential(sigma_mean)  # Distributions.jl: Exponential(scale); NumPyro: Exponential(rate) → scale = 1/rate
    mu = alpha .+ X * beta
    Turing.@addlogprob! sum(logpdf.(LocationScale.(mu, sigma, TDist(nu)), Y))
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
    Y = Float64.(data["Y"])

    model = turing_robust_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc, sigma_mean)

    return (
        turing_model = model,
        model_name = "Robust Regression",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    alpha = Array(chain[:alpha])[start:total]
    k = length([s for s in names(chain, :parameters) if startswith(string(s), "beta[")])
    beta = hcat([Array(chain[Symbol("beta[$i]")])[start:total] for i in 1:k]...)
    nu = Array(chain[:nu])[start:total]
    sigma = Array(chain[:sigma])[start:total]

    return Dict{String,Any}(
        "alpha" => collect(alpha),
        "beta" => [collect(beta[i, :]) for i in axes(beta, 1)],
        "nu" => collect(nu),
        "sigma" => collect(sigma),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    beta = Float64.(init_params["beta"])
    nu = Float64(init_params["nu"][1])
    sigma = Float64(init_params["sigma"][1])
    return (alpha=alpha, beta=beta, nu=nu, sigma=sigma)
end
