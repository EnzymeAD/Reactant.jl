# Turing.jl model for GP with Fixed Lengthscale
# y ~ MVN(0, alpha^2 * K_base + sigma^2 * I)

using Turing, Distributions, LinearAlgebra

@model function turing_gp_fixed_lengthscale(K_base, Y, n)
    alpha ~ truncated(Normal(0.0, 1.0); lower=0.0)
    sigma ~ truncated(Normal(0.0, 1.0); lower=0.0)

    cov = alpha^2 * K_base + sigma^2 * I(n)
    F = cholesky(Symmetric(cov), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    Turing.@addlogprob! -0.5 * (dot(Y, F \ Y) + logdet(F) + n * log(2π))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    rho = Float64(attrs["rho"])

    X = Float64.(data["X"])
    Y = Float64.(data["Y"])

    # Precompute SE kernel (sample-invariant)
    K_base = [exp(-0.5 * ((X[i] - X[j]) / rho)^2) for i in 1:n, j in 1:n]

    model = turing_gp_fixed_lengthscale(K_base, Y, n)
    return (turing_model = model, model_name = "GP Fixed Lengthscale")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1
    return Dict{String,Any}(
        "alpha" => collect(Array(chain[:alpha])[start:total]),
        "sigma" => collect(Array(chain[:sigma])[start:total]),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    sigma = Float64(init_params["sigma"][1])
    return (alpha=alpha, sigma=sigma)
end
