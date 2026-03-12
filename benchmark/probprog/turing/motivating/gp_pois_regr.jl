# Turing.jl model spec for GP Poisson Regression (posteriordb gp_pois_regr)
# Non-centered parameterization, fixed rho, no jitter.
# Matches NumPyro gp_pois_regr.py and Impulse motivating/gp_pois_regr.jl

using Turing, Distributions, LinearAlgebra

@model function turing_gp_pois_regr(x, k_obs, rho, n)
    alpha ~ truncated(Normal(0.0, 2.0), 0.0, Inf)  # HalfNormal(2.0)
    f_tilde ~ filldist(Normal(0.0, 1.0), n)

    # SE kernel with fixed rho (no jitter)
    delta = reshape(x, n, 1) .- reshape(x, 1, n)
    K_base = exp.(-0.5 .* (delta ./ rho) .^ 2)

    # NCP: cov = alpha^2 * K_base, f = chol(cov) @ f_tilde
    cov = alpha^2 .* K_base
    F = cholesky(Symmetric(cov), check=false)
    if !issuccess(F)
        Turing.@addlogprob! -Inf
        return
    end
    f = F.L * f_tilde

    Turing.@addlogprob! sum(logpdf.(Poisson.(exp.(f)), k_obs))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    rho = Float64(attrs["rho"])

    x = Float64.(data["X"])
    k_obs = Float64.(data["Y"])

    model = turing_gp_pois_regr(x, k_obs, rho, n)

    return (
        turing_model = model,
        model_name = "GP Poisson Regression",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    alpha = Array(chain[:alpha])[start:total]
    n = length([s for s in names(chain, :parameters) if startswith(string(s), "f_tilde[")])
    f_tilde = hcat([Array(chain[Symbol("f_tilde[$i]")])[start:total] for i in 1:n]...)

    return Dict{String,Any}(
        "alpha" => collect(alpha),
        "f_tilde" => [collect(f_tilde[i, :]) for i in axes(f_tilde, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    # alpha is HalfNormal — init_params are in unconstrained (log) space
    alpha = exp(Float64(init_params["alpha"][1]))
    f_tilde = Float64.(init_params["f_tilde"])
    return (alpha=alpha, f_tilde=f_tilde)
end
