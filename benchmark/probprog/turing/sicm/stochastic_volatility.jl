# Turing.jl model for Stochastic Volatility
# h = mu + chol(sigma_h^2 * C_ar1) * eta, y_t ~ Normal(0, exp(h_t / 2))

using Turing, Distributions, LinearAlgebra

@model function turing_stochastic_vol(C_ar1, Y, T)
    mu ~ Normal(0.0, 5.0)
    sigma_h ~ truncated(Normal(0.0, 1.0); lower=0.0)
    eta ~ filldist(Normal(0.0, 1.0), T)

    cov_h = sigma_h^2 * C_ar1
    F = cholesky(Symmetric(cov_h), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    h = mu .+ F.L * eta

    Turing.@addlogprob! sum(logpdf.(Normal.(0.0, exp.(h ./ 2)), Y))
end

function setup(data)
    attrs = data["attrs"]
    T = Int(attrs["T"])

    C_ar1_raw = data["C_ar1"]
    C_ar1 = [Float64(C_ar1_raw[i][j]) for i in 1:T, j in 1:T]

    Y = Float64.(data["Y"])

    model = turing_stochastic_vol(C_ar1, Y, T)
    return (turing_model = model, model_name = "Stochastic Volatility")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    ki = 1
    while Symbol("eta[$ki]") in chain.name_map.parameters
        ki += 1
    end
    T = ki - 1

    eta_samples = hcat([Array(chain[Symbol("eta[$i]")])[start:total] for i in 1:T]...)

    return Dict{String,Any}(
        "mu" => collect(Array(chain[:mu])[start:total]),
        "sigma_h" => collect(Array(chain[:sigma_h])[start:total]),
        "eta" => [collect(eta_samples[i, :]) for i in 1:size(eta_samples, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    mu = Float64(init_params["mu"][1])
    sigma_h = Float64(init_params["sigma_h"][1])
    eta = Float64.(init_params["eta"])
    return (mu=mu, sigma_h=sigma_h, eta=eta)
end
