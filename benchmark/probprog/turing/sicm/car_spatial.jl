# Turing.jl model for CAR Spatial
# phi ~ MVN(0, sigma_phi^2 * Q_inv), y ~ Poisson(exp(log_E + alpha + phi))
# Uses CENTERED parameterization (matching NumPyro, Impulse, Stan)

using Turing, Distributions, LinearAlgebra

@model function turing_car_spatial(Q_inv, log_E, Y, n)
    alpha ~ Normal(0.0, 10.0)
    sigma_phi ~ truncated(Normal(0.0, 1.0); lower=0.0)

    Sigma_phi = Symmetric(sigma_phi^2 * Q_inv)
    phi ~ MvNormal(zeros(n), Sigma_phi)

    log_rates = log_E .+ alpha .+ phi
    Turing.@addlogprob! sum(logpdf.(Poisson.(exp.(log_rates)), Y))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    Q_inv_raw = data["Q_inv"]
    Q_inv = [Float64(Q_inv_raw[i][j]) for i in 1:n, j in 1:n]

    log_E = Float64.(data["log_E"])
    Y_raw = data["Y"]
    Y = [Int(round(Float64(Y_raw[i]))) for i in 1:n]

    model = turing_car_spatial(Q_inv, log_E, Y, n)
    return (turing_model = model, model_name = "CAR Spatial")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    ki = 1
    while Symbol("phi[$ki]") in chain.name_map.parameters
        ki += 1
    end
    n = ki - 1

    phi_samples = hcat([Array(chain[Symbol("phi[$i]")])[start:total] for i in 1:n]...)

    return Dict{String,Any}(
        "alpha" => collect(Array(chain[:alpha])[start:total]),
        "sigma_phi" => collect(Array(chain[:sigma_phi])[start:total]),
        "phi" => [collect(phi_samples[i, :]) for i in 1:size(phi_samples, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    sigma_phi = Float64(init_params["sigma_phi"][1])
    phi = Float64.(init_params["phi"])
    return (alpha=alpha, sigma_phi=sigma_phi, phi=phi)
end
