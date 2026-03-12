# Turing.jl model spec for Gaussian Process regression
# Matches NumPyro gaussian_process.py and Impulse standard/gaussian_process.jl

using Turing, Distributions, LinearAlgebra

@model function turing_gp(X, Y, n)
    kernel_var ~ LogNormal(0.0, 10.0)
    kernel_length ~ LogNormal(0.0, 10.0)
    kernel_noise ~ LogNormal(0.0, 10.0)

    # Squared exponential kernel (matching NumPyro/Impulse)
    delta = reshape(X, n, 1) .- reshape(X, 1, n)
    deltaXsq = (delta ./ kernel_length) .^ 2
    K = kernel_var .* exp.(-0.5 .* deltaXsq)
    K = K + (kernel_noise + 1.0e-6) * I

    # Manual Cholesky with check=false to handle non-PD proposals gracefully.
    # JAX/NumPyro returns NaN for non-PD Cholesky (→ NUTS divergence/reject);
    # Julia throws PosDefException by default. Use check=false + issuccess to
    # return -Inf logprob instead, matching the JAX behavior.
    F = cholesky(Symmetric(K), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    # Manual MvNormal logpdf: -0.5 * (y'K^{-1}y + log|K| + n*log(2π))
    Turing.@addlogprob! -0.5 * (dot(Y, F \ Y) + logdet(F) + n * log(2π))
end

function setup(data)
    n = Int(data["attrs"]["n"])
    X = Float64.(data["X"])
    Y = Float64.(data["Y"])

    model = turing_gp(X, Y, n)

    return (
        turing_model = model,
        model_name = "Gaussian Process",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    return Dict{String,Any}(
        "kernel_var" => collect(Array(chain[:kernel_var])[start:total]),
        "kernel_length" => collect(Array(chain[:kernel_length])[start:total]),
        "kernel_noise" => collect(Array(chain[:kernel_noise])[start:total]),
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    # init_params are in constrained (positive) space — no exp() needed
    kernel_var = Float64(init_params["kernel_var"][1])
    kernel_length = Float64(init_params["kernel_length"][1])
    kernel_noise = Float64(init_params["kernel_noise"][1])
    return (kernel_var=kernel_var, kernel_length=kernel_length, kernel_noise=kernel_noise)
end
