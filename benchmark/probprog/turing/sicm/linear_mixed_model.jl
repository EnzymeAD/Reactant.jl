# Turing.jl model for Linear Mixed Model
# y ~ MVN(X * beta, sigma_u^2 * K_re + sigma_e^2 * I)

using Turing, Distributions, LinearAlgebra

@model function turing_linear_mixed(X, K_re, Y, n, k)
    beta ~ filldist(Normal(0.0, 1.0), k)
    sigma_u ~ truncated(Normal(0.0, 1.0); lower=0.0)
    sigma_e ~ truncated(Normal(0.0, 1.0); lower=0.0)

    mu = X * beta
    cov = sigma_u^2 * K_re + sigma_e^2 * I(n)
    F = cholesky(Symmetric(cov), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    r = Y - mu
    Turing.@addlogprob! -0.5 * (dot(r, F \ r) + logdet(F) + n * log(2π))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]

    K_re_raw = data["K_re"]
    K_re = [Float64(K_re_raw[i][j]) for i in 1:n, j in 1:n]

    Y = Float64.(data["Y"])

    model = turing_linear_mixed(X, K_re, Y, n, k)
    return (turing_model = model, model_name = "Linear Mixed Model")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    ki = 1
    while Symbol("beta[$ki]") in chain.name_map.parameters
        ki += 1
    end
    k = ki - 1

    beta_samples = hcat([Array(chain[Symbol("beta[$i]")])[start:total] for i in 1:k]...)

    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma_u" => collect(Array(chain[:sigma_u])[start:total]),
        "sigma_e" => collect(Array(chain[:sigma_e])[start:total]),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    beta = Float64.(init_params["beta"])
    sigma_u = Float64(init_params["sigma_u"][1])
    sigma_e = Float64(init_params["sigma_e"][1])
    return (beta=beta, sigma_u=sigma_u, sigma_e=sigma_e)
end
