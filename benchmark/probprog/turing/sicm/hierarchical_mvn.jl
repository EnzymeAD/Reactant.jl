# Turing.jl model for Hierarchical MVN
# Sigma = diag(sigma_k) * Omega * diag(sigma_k)
# theta_j = mu + chol(Sigma) * eta_j
# y_{jk} ~ Normal(theta_{jk}, sigma_obs)

using Turing, Distributions, LinearAlgebra

@model function turing_hier_mvn(Omega, Y, K, J)
    mu ~ filldist(Normal(0.0, 10.0), K)
    sigma_k ~ filldist(truncated(Normal(0.0, 1.0); lower=0.0), K)
    sigma_obs ~ truncated(Normal(0.0, 1.0); lower=0.0)
    eta ~ filldist(Normal(0.0, 1.0), K * J)

    # Covariance: diag(sigma_k) * Omega * diag(sigma_k)
    D = Diagonal(sigma_k)
    Sigma = D * Omega * D

    F = cholesky(Symmetric(Sigma), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    L = F.L

    # NCP: theta_j = mu + L * eta_j
    eta_matrix = reshape(eta, K, J)
    theta = mu .+ L * eta_matrix  # K x J

    Turing.@addlogprob! sum(logpdf.(Normal.(theta, sigma_obs), Y))
end

function setup(data)
    attrs = data["attrs"]
    K = Int(attrs["K"])
    J = Int(attrs["J"])

    Omega_raw = data["Omega"]
    Omega = [Float64(Omega_raw[i][j]) for i in 1:K, j in 1:K]

    Y_raw = data["Y"]
    Y = [Float64(Y_raw[k][j]) for k in 1:K, j in 1:J]

    model = turing_hier_mvn(Omega, Y, K, J)

    return (
        turing_model = model,
        model_name = "Hierarchical MVN",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    # Determine K from mu symbols
    ki = 1
    while Symbol("mu[$ki]") in chain.name_map.parameters
        ki += 1
    end
    K = ki - 1

    # Determine J*K from eta symbols
    ei = 1
    while Symbol("eta[$ei]") in chain.name_map.parameters
        ei += 1
    end
    n_eta = ei - 1

    mu_samples = hcat([Array(chain[Symbol("mu[$i]")])[start:total] for i in 1:K]...)
    sigma_k_samples = hcat([Array(chain[Symbol("sigma_k[$i]")])[start:total] for i in 1:K]...)
    eta_samples = hcat([Array(chain[Symbol("eta[$i]")])[start:total] for i in 1:n_eta]...)

    return Dict{String,Any}(
        "mu" => [collect(mu_samples[i, :]) for i in 1:size(mu_samples, 1)],
        "sigma_k" => [collect(sigma_k_samples[i, :]) for i in 1:size(sigma_k_samples, 1)],
        "sigma_obs" => collect(Array(chain[:sigma_obs])[start:total]),
        "eta" => [collect(eta_samples[i, :]) for i in 1:size(eta_samples, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    mu = Float64.(init_params["mu"])
    sigma_k = Float64.(init_params["sigma_k"])
    sigma_obs = Float64(init_params["sigma_obs"][1])
    eta = Float64.(init_params["eta"])
    return (mu=mu, sigma_k=sigma_k, sigma_obs=sigma_obs, eta=eta)
end
