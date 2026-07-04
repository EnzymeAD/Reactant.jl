# Turing.jl model for Phylogenetic Regression
# y ~ MVN(X * beta, sigma_p^2 * C_phylo + sigma_e^2 * C_env)

using Turing, Distributions, LinearAlgebra

@model function turing_phylo_regr(X, C_phylo, C_env, Y, n, k)
    beta ~ filldist(Normal(0.0, 10.0), k)
    sigma_p ~ truncated(Normal(0.0, 1.0); lower=0.0)
    sigma_e ~ truncated(Normal(0.0, 1.0); lower=0.0)

    mu = X * beta
    K = sigma_p^2 * C_phylo + sigma_e^2 * C_env

    F = cholesky(Symmetric(K), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    Turing.@addlogprob! -0.5 * (dot(Y - mu, F \ (Y - mu)) + logdet(F) + n * log(2π))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]

    C_phylo_raw = data["C_phylo"]
    C_phylo = [Float64(C_phylo_raw[i][j]) for i in 1:n, j in 1:n]

    C_env_raw = data["C_env"]
    C_env = [Float64(C_env_raw[i][j]) for i in 1:n, j in 1:n]

    Y = Float64.(data["Y"])

    model = turing_phylo_regr(X, C_phylo, C_env, Y, n, k)

    return (
        turing_model = model,
        model_name = "Phylogenetic Regression",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    k = size(Array(chain[Symbol("beta[1]")]), 1) > 0 ? 0 : 0
    # Determine k from available beta symbols
    ki = 1
    while Symbol("beta[$ki]") in chain.name_map.parameters
        ki += 1
    end
    k = ki - 1

    beta_samples = hcat([Array(chain[Symbol("beta[$i]")])[start:total] for i in 1:k]...)

    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma_p" => collect(Array(chain[:sigma_p])[start:total]),
        "sigma_e" => collect(Array(chain[:sigma_e])[start:total]),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    beta = Float64.(init_params["beta"])
    sigma_p = Float64(init_params["sigma_p"][1])
    sigma_e = Float64(init_params["sigma_e"][1])
    return (beta=beta, sigma_p=sigma_p, sigma_e=sigma_e)
end
