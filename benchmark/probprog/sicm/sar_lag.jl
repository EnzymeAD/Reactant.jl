# Bayesian spatial-lag model (SAR)  -- SICM: DotScale (+ EigenLift) showcase
#
# Model (Anselin 1988; LeSage & Pace 2009):
#   (I - rho W) y = X beta + eps,   eps ~ N(0, sigma^2 I)
#   rho_raw ~ Normal(0, 1)          rho = lo + (hi - lo) * logistic(rho_raw)
#   beta    ~ Normal(0, 10)^k
#   sigma   ~ HalfNormal(1)
#
# W is a KNOWN contiguity matrix, so it is DATA. The log-likelihood is
#
#   log p(y) = log|I - rho W| - n log sigma - ||A y - X beta||^2 / (2 sigma^2)
#              - (n/2) log(2 pi),      A y = y - (rho W) y
#
# Two SICM targets, written here exactly as a naive translation would:
#
#  1. DotScale. The lag term is spelled `(rho .* W) * y`, i.e. the scalar is
#     pushed into the matrix and then the matrix multiplies the response.
#     DotGeneralScaleFactorization rewrites dot(s*W, y) -> s * dot(W, y); since
#     W and y are both data, `W y` is then fully sample-invariant and hoists
#     out of the leapfrog step, turning an O(n^2) matvec per step into a
#     scalar rescale of a precomputed vector.
#
#  2. EigenLift. The log-determinant is taken as a Cholesky of I - rho Ws,
#     which has the shape 1*I + (-rho)*Ws with Ws invariant, i.e. exactly the
#     CholeskyEigenLift structure; the hoisted eigendecomposition of Ws turns
#     an O(n^3) factorization per step into sum(log(1 - rho lambda_i)).
#
# This row is therefore honestly a DotScale + EigenLift cascade; the results
# file records which one the IR shows dominating.
#
# Two weight matrices, both data (see real_data/README.md):
#   W  = D^-1 B   row-standardised contiguity, the lag operator of the model
#                 equation and the convention the literature reports rho in;
#   Ws = D^-1/2 B D^-1/2   symmetric normalised adjacency, similar to W and so
#                 of identical spectrum, used for the determinant because the
#                 Cholesky needs a symmetric positive definite matrix. The
#                 identity log|I - rho W| = log|I - rho Ws| is exact (Ord 1975;
#                 spdep's `similar.listw`) and was checked numerically when the
#                 data file was built.
#
# Data: Boston housing, 506 census tracts (Harrison & Rubinfeld 1978, with
# Gilley & Pace's corrected CMEDV) with the sphere-of-influence contiguity,
# from the R package spData.

function _sar_sampler(rng, rho_raw, beta, sigma, W, Ws, X, y, n, k, lo, hi)
    return Reactant.Ops.fill(0.0, n)
end

function _sar_logpdf(_obs, rho_raw, beta, sigma, W, Ws, X, y, n, k, lo, hi)
    # rho constrained to the stationary interval by a scaled logistic. Kept as a
    # 1-element array throughout: scalar indexing of a traced array is disallowed.
    rho1 = lo .+ (hi - lo) ./ (1.0 .+ exp.(.-rho_raw))
    r2d = reshape(rho1, 1, 1)

    # Lag term, naive spelling: scale the matrix, then multiply the response.
    # -> DotGeneralScaleFactorization, then `W * y` is invariant and hoists.
    rW = r2d .* W
    lag = rW * reshape(y, n, 1)
    Ay = reshape(y, n, 1) .- lag

    resid = Ay .- X * reshape(beta, k, 1)
    quad = sum(resid .* resid)

    # log|I - rho Ws| : shape 1*I + (-rho)*Ws  -> CholeskyEigenLift.
    # I_n is built here rather than passed in so that tracing folds it into a
    # dense constant, which is the form the pattern's diagonal-shift matcher
    # recognises (same as sicm/linear_mixed_model.jl).
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    A = I_n .+ ((-1.0) .* r2d) .* Ws
    LA = Reactant.Ops.cholesky(A; lower=true)
    logdetA = 2.0 * sum(log.(LinearAlgebra.diag(LA)))

    per = logdetA .- n .* log.(sigma) .- 0.5 .* quad ./ (sigma .* sigma)
    return sum(per) - 0.5 * n * log(2 * pi)
end

function model_sar(rng, W, Ws, X, y, n, k, lo, hi)
    _, rho_raw = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:rho_raw
    )
    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (k,)); symbol=:beta
    )
    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )
    _, _ = ProbProg.sample(
        rng, _sar_sampler, rho_raw, beta, sigma, W, Ws, X, y, n, k, lo, hi;
        symbol=:y, logpdf=_sar_logpdf,
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    lo = Float64(attrs["rho_lo"])
    hi = Float64(attrs["rho_hi"])

    W_raw = data["W"]
    W = [Float64(W_raw[i][j]) for i in 1:n, j in 1:n]
    Ws_raw = data["Ws"]
    Ws = [Float64(Ws_raw[i][j]) for i in 1:n, j in 1:n]
    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    y = Float64.(data["Y"])

    model_args = (
        Reactant.to_rarray(W),
        Reactant.to_rarray(Ws),
        Reactant.to_rarray(X),
        Reactant.to_rarray(y),
        n,
        k,
        lo,
        hi,
    )

    selection = ProbProg.select(
        ProbProg.Address(:rho_raw),
        ProbProg.Address(:beta),
        ProbProg.Address(:sigma),
    )

    return (
        model_fn = model_sar,
        model_args = model_args,
        selection = selection,
        position_size = k + 2,
        model_name = "SAR spatial lag (Boston)",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_rho_raw = Float64.(init_params["rho_raw"])
        init_beta = Float64.(init_params["beta"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_rho_raw = [0.0]
        init_beta = zeros(k)
        init_sigma = [1.0]
    end

    return ProbProg.Constraint(
        :rho_raw => init_rho_raw,
        :beta => init_beta,
        :sigma => init_sigma,
        :y => y,
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "rho_raw" => collect(vec(trace.choices[:rho_raw])),
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
