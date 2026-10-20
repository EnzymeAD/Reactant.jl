# Turing.jl model for the Brownian-motion phylogenetic regression of primate brain size
# McElreath, Statistical Rethinking, 2nd ed. (2020), section 14.5.2, model m14.10, as the
# book writes it: the covariance R * sigma_sq goes to the built-in MvNormal, which factors
# it at every evaluation.
#   B ~ MvNormal(a + bM * M + bG * G, R * sigma_sq)

using Turing, Distributions, LinearAlgebra

@model function turing_primates_brownian(M, G, R, B)
    a ~ Normal(0.0, 1.0)
    bM ~ Normal(0.0, 0.5)
    bG ~ Normal(0.0, 0.5)
    sigma_sq ~ Exponential(1.0)

    mu = a .+ bM .* M .+ bG .* G
    SIGMA = R .* sigma_sq

    B ~ MvNormal(mu, SIGMA)
end

function setup(data)
    n = Int(data["attrs"]["n"])

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    X_raw = data["X"]
    M = [Float64(X_raw[i][2]) for i in 1:n]
    G = [Float64(X_raw[i][3]) for i in 1:n]
    B = Float64.(data["Y"])

    model = turing_primates_brownian(M, G, R, B)

    return (
        turing_model = model,
        model_name = "Brownian motion model m14.10 (Primates301)",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1
    return Dict{String,Any}(
        name => collect(Array(chain[Symbol(name)])[start:total])
        for name in ("a", "bM", "bG", "sigma_sq")
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    return (
        a=Float64(init_params["a"][1]),
        bM=Float64(init_params["bM"][1]),
        bG=Float64(init_params["bG"][1]),
        sigma_sq=Float64(init_params["sigma_sq"][1]),
    )
end
