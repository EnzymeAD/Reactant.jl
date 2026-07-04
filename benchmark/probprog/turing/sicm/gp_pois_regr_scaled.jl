# Turing.jl model for GP Poisson Regression (Scaled)
# f = chol(alpha^2 * K_base) * f_tilde, k ~ Poisson(exp(f))

using Turing, Distributions, LinearAlgebra

@model function turing_gp_pois_regr_scaled(K_base, Y, n)
    alpha ~ truncated(Normal(0.0, 2.0); lower=0.0)
    f_tilde ~ filldist(Normal(0.0, 1.0), n)

    cov = alpha^2 * K_base
    F = cholesky(Symmetric(cov), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    f = F.L * f_tilde

    Turing.@addlogprob! sum(logpdf.(Poisson.(exp.(f)), Y))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    rho = Float64(attrs["rho"])

    X = Float64.(data["X"])
    Y_raw = data["Y"]
    Y = [Int(round(Float64(Y_raw[i]))) for i in 1:n]

    # Precompute SE kernel (sample-invariant)
    K_base = [exp(-0.5 * ((X[i] - X[j]) / rho)^2) for i in 1:n, j in 1:n]

    model = turing_gp_pois_regr_scaled(K_base, Y, n)
    return (turing_model = model, model_name = "GP Pois. Regr. (scaled)")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    ki = 1
    while Symbol("f_tilde[$ki]") in chain.name_map.parameters
        ki += 1
    end
    n = ki - 1

    f_tilde_samples = hcat([Array(chain[Symbol("f_tilde[$i]")])[start:total] for i in 1:n]...)

    return Dict{String,Any}(
        "alpha" => collect(Array(chain[:alpha])[start:total]),
        "f_tilde" => [collect(f_tilde_samples[i, :]) for i in 1:size(f_tilde_samples, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    f_tilde = Float64.(init_params["f_tilde"])
    return (alpha=alpha, f_tilde=f_tilde)
end
