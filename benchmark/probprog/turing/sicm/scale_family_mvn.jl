# Turing.jl model for Scale-Family MVN
# y ~ MVN(0, tau^2 * R)

using Turing, Distributions, LinearAlgebra

@model function turing_scale_family_mvn(R, Y, n)
    tau ~ truncated(Normal(0.0, 1.0); lower=0.0)

    cov = tau^2 * R
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

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    Y = Float64.(data["Y"])

    model = turing_scale_family_mvn(R, Y, n)
    return (turing_model = model, model_name = "Scale-Family MVN")
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1
    return Dict{String,Any}(
        "tau" => collect(Array(chain[:tau])[start:total]),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    tau = Float64(init_params["tau"][1])
    return (tau=tau,)
end
