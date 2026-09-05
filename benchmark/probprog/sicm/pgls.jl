# Phylogenetic Generalized Least Squares (PGLS)  -- SICM: TriSolveScale showcase
#
# Model (Grafen 1989; the caper vignette's simple PGLS):
#   y ~ MVN(X * beta, sigma^2 * C)
#   beta  ~ Normal(0, 10)^k
#   sigma ~ HalfNormal(1)
#   C = phylogenetic correlation of a KNOWN tree under Brownian motion
#       (ape::vcv.phylo, lambda = 1), so C is DATA and sample-invariant.
#
# Why this row is a TriangularSolveScale showcase rather than another CholScale
# row. The density is written in the whitened-Cholesky form of GLS: whiten the
# response and the design matrix separately against the Cholesky factor of the
# covariance, then take an ordinary least-squares residual in the whitened
# space,
#
# This is one of the two standard GLS spellings, and the numerically preferred
# one; it is what Stan's `multi_normal_cholesky` and brms's phylogenetic models
# use. NOTE it is NOT what caper's own `pgls` does: caper forms the explicit
# inverse `iV <- solve(V)` and accumulates `X' iV X` and `X' iV y`
# (caper/R/pgls.R:383-393), using a whitening matrix only for phylogenetic
# residuals. The model fitted here is the vignette's model; the spelling of its
# density is our choice, and it is the choice that exposes the pattern.
#
#   L      = chol(sigma^2 * C)
#   ytilde = L \ y            <- n x 1 solve, RIGHT-HAND SIDE IS DATA
#   Xtilde = L \ X            <- n x k solve, RIGHT-HAND SIDE IS DATA
#   r      = ytilde - Xtilde * beta
#   log p  = -r'r/2 - sum(log diag(L)) - (n/2) log(2 pi)
#
# In the CAR rows the density's solves act on a SAMPLED vector (the spatial
# field phi), so no scale rewrite can hoist them and they survive in both arms.
# Here both solve right-hand sides are data, so the cascade
#
#   CholeskyScaleFactorization:     chol(sigma^2 C)      -> sigma * chol(C)
#   TriangularSolveScaleFactorization: solve(sigma*L_C, D) -> (1/sigma) * solve(L_C, D)
#   LogMultiplyDistribution:        log diag(sigma*L_C)  -> log sigma + log diag(L_C)
#
# leaves `solve(L_C, y)` and `solve(L_C, X)` fully sample-invariant, so the
# SOLVES THEMSELVES hoist out of the leapfrog step, not just the factorization.
# The expected IR signature is therefore both the n x 1 and the n x k
# triangular solve leaving the loop.
#
# Data: the `shorebird` data of the R package caper (71 species; traits from
# Lislevand & Thomas 2006/2007), mirroring the vignette model
# pgls(log(Egg.Mass) ~ log(M.Mass), shorebird) at lambda = 1.
# See pplbench/pplbench/models/real_data/README.md.

# Placeholder sampler for the observed site: the value is supplied by the
# constraint and the density enters through `_pgls_logpdf`, which reads the
# data from the model arguments (same convention as
# standard/crowd_sourced_annotation.jl).
function _pgls_sampler(rng, beta, sigma, C, X, y, n, k)
    return Reactant.Ops.fill(0.0, n)
end

function _pgls_logpdf(_obs, beta, sigma, C, X, y, n, k)
    s2d = reshape(sigma, 1, 1)

    # Covariance sigma^2 * C  ->  CholeskyScaleFactorization
    Sigma = (s2d .* s2d) .* C
    L = Reactant.Ops.cholesky(Sigma; lower=true)      # Sigma = L * L'

    # Whitening solves; BOTH right-hand sides are data.
    Lo = LinearAlgebra.LowerTriangular(L)
    ytilde = Lo \ reshape(y, n, 1)                    # n x 1
    Xtilde = Lo \ X                                   # n x k

    r = ytilde .- Xtilde * reshape(beta, k, 1)
    quad = sum(r .* r)

    # 0.5 * log det(Sigma) = sum(log(diag(L)))  ->  LogMultiplyDistribution
    half_logdet = sum(log.(LinearAlgebra.diag(L)))

    return -0.5 * quad - half_logdet - 0.5 * n * log(2 * pi)
end

function model_pgls(rng, C, X, y, n, k)
    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (k,)); symbol=:beta
    )
    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )
    _, _ = ProbProg.sample(
        rng, _pgls_sampler, beta, sigma, C, X, y, n, k;
        symbol=:y, logpdf=_pgls_logpdf,
    )
    return nothing
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

    model_args = (
        Reactant.to_rarray(C),
        Reactant.to_rarray(X),
        Reactant.to_rarray(y),
        n,
        k,
    )

    selection = ProbProg.select(
        ProbProg.Address(:beta),
        ProbProg.Address(:sigma),
    )

    return (
        model_fn = model_pgls,
        model_args = model_args,
        selection = selection,
        position_size = k + 1,
        model_name = "PGLS (shorebird)",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_beta = zeros(k)
        init_sigma = [1.0]
    end

    return ProbProg.Constraint(
        :beta => init_beta,
        :sigma => init_sigma,
        :y => y,
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
