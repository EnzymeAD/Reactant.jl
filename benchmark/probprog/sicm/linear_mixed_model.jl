# Linear Mixed Model (SICM: CholeskyEigenLift)
#
# Model: y ~ MVN(X * beta, sigma_u^2 * K_re + sigma_e^2 * I)
#   beta ~ Normal(0, 1)^k
#   sigma_u ~ HalfNormal(1)    (random effects std)
#   sigma_e ~ HalfNormal(1)    (residual std)
#   K_re = Z * Z'              (precomputed, sample-invariant)
#   X is fixed design matrix, Z is random effects design matrix
#
# SICM EigenLift pattern:
#   cholesky(s * K_re + t * I) -> eigendecomposition of K_re (hoisted, O(N^3))
#   Per-iteration: O(N^2) matvec instead of O(N^3) Cholesky
#
# Motivation: genetics (kinship matrix), education (school effects),
#   spatial random effects with known structure.

function model_lmm(rng, X, K_re, y)
    n = Base.length(y)
    k = size(X, 2)

    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (k,)); symbol=:beta
    )

    _, sigma_u = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_u
    )

    _, sigma_e = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_e
    )

    # Mean: X * beta
    mu = reshape(X * reshape(beta, k, 1), n)

    # Covariance: sigma_u^2 * K_re + sigma_e^2 * I
    su_2d = reshape(sigma_u, 1, 1)
    se_2d = reshape(sigma_e, 1, 1)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    K = (su_2d .* su_2d) .* K_re .+ (se_2d .* se_2d) .* I_n

    _, obs = ProbProg.sample(
        rng, ProbProg.MultiNormal(mu, K, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    q = Int(attrs["q"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    K_raw = data["K_re"]
    K_re = [Float64(K_raw[i][j]) for i in 1:n, j in 1:n]
    y = Float64.(data["Y"])

    X_rarray = Reactant.to_rarray(X)
    K_re_rarray = Reactant.to_rarray(K_re)
    model_args = (X_rarray, K_re_rarray, y)

    selection = ProbProg.select(
        ProbProg.Address(:beta),
        ProbProg.Address(:sigma_u),
        ProbProg.Address(:sigma_e),
    )

    return (
        model_fn = model_lmm,
        model_args = model_args,
        selection = selection,
        position_size = k + 2,
        model_name = "Linear Mixed Model",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma_u = Float64.(init_params["sigma_u"])
        init_sigma_e = Float64.(init_params["sigma_e"])
    else
        init_beta = zeros(k)
        init_sigma_u = [1.0]
        init_sigma_e = [1.0]
    end

    return ProbProg.Constraint(
        :beta => init_beta,
        :sigma_u => init_sigma_u,
        :sigma_e => init_sigma_e,
        :y => y,
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma_u" => collect(vec(trace.choices[:sigma_u])),
        "sigma_e" => collect(vec(trace.choices[:sigma_e])),
    )
end
