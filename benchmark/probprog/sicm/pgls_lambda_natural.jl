# PGLS with Pagel's lambda estimated, written as the model is written
#
# Model (Pagel 1999; Freckleton, Harvey & Pagel 2002), the Bayesian form of the
# caper vignette's `pgls(log(Egg.Mass) ~ log(M.Mass), shorebird, lambda = 'ML')`:
#   y ~ MultivariateNormal(X * beta, sigma^2 * (lambda * C + (1 - lambda) * I))
#   beta   ~ Normal(0, 10)^k
#   sigma  ~ HalfNormal(1)
#   lambda ~ Uniform(0, 1)
#   C = Brownian-motion phylogenetic correlation of the KNOWN tree (DATA), normalised
#       to unit root height (diag C = 1), so lambda * C + (1 - lambda) * I is Pagel's
#       transform (off-diagonals scaled by lambda, diagonal kept).
#
# Spelling. The mean and the covariance are formed as the model states them and the
# response is scored by the front-end MultivariateNormal on (X * beta, Sigma), whose
# emitted density factors Sigma and solves against the factor at every evaluation
# (the density Stan's multi_normal computes). No whitening, no hand-written
# log-determinant: the same row as sicm/pgls_lambda.jl, whose whitened-GLS spelling
# (separate L \ y and L \ X, an OLS residual, a masked diag(L)) is an expert rewrite
# of this one. Same priors, data and parameterization: beta and sigma as there,
# lambda through the same unit-interval site (HMC samples logit(lambda) with the
# sigmoid Jacobian; ProbProg.Uniform is not an HMC/NUTS reparameterization target).
#
# Data: caper `shorebird` (71 species; Lislevand & Thomas 2006), the same file as
# sicm/pgls_lambda.jl (ablation/sicm_roster/data/pgls_lambda.json).

# lambda ~ Uniform(0, 1); `support=:unit_interval` makes HMC sample logit(lambda)
# with the sigmoid Jacobian (NumPyro's dist.Uniform(0, 1)).
function _unit_uniform_sampler(rng)
    return rand(rng, 1)
end

function _unit_uniform_logpdf(lambda)
    return sum(0.0 .* lambda)
end

function model_pgls_lambda_natural(rng, C, X, y, n, k)
    _, beta = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (k,)); symbol=:beta)
    _, sigma = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma)
    _, lambda = ProbProg.sample(
        rng,
        _unit_uniform_sampler;
        symbol=:lambda,
        logpdf=_unit_uniform_logpdf,
        support=:unit_interval,
    )

    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    mu = X * beta
    Sigma = sigma .^ 2 .* (lambda .* C .+ (1 .- lambda) .* I_n)

    _, y = ProbProg.sample(rng, ProbProg.MultivariateNormal(mu, Sigma, (n,)); symbol=:y)
    return y
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    C_raw = data["C"]
    C = [Float64(C_raw[i][j]) for i in 1:n, j in 1:n]
    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    y = Float64.(data["Y"])

    model_args = (Reactant.to_rarray(C), Reactant.to_rarray(X), Reactant.to_rarray(y), n, k)

    selection = ProbProg.select(
        ProbProg.Address(:beta), ProbProg.Address(:sigma), ProbProg.Address(:lambda)
    )

    return (
        model_fn=model_pgls_lambda_natural,
        model_args=model_args,
        selection=selection,
        position_size=k + 2,
        model_name="PGLS, Pagel's lambda estimated (shorebird, natural MVN)",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma = Float64.(init_params["sigma"])
        init_lambda = Float64.(init_params["lambda"])
    else
        init_beta = zeros(k)
        init_sigma = [1.0]
        init_lambda = [0.5]
    end

    return ProbProg.Constraint(
        :beta => init_beta, :sigma => init_sigma, :lambda => init_lambda, :y => y
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma" => collect(vec(trace.choices[:sigma])),
        "lambda" => collect(vec(trace.choices[:lambda])),
    )
end
