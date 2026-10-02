# PGLS with Pagel's lambda ESTIMATED  -- SICM: EigenLift
#
# Model (Pagel 1999; Freckleton, Harvey & Pagel 2002), the Bayesian form of the
# caper vignette's `pgls(log(Egg.Mass) ~ log(M.Mass), shorebird, lambda = 'ML')`:
#   y ~ MVN(X * beta, sigma^2 * V(lambda))
#   V(lambda) = lambda * C + (1 - lambda) * I
#   beta   ~ Normal(0, 10)^k
#   sigma  ~ HalfNormal(1)
#   lambda ~ Uniform(0, 1)
#   C = Brownian-motion phylogenetic correlation of the KNOWN tree (DATA).
#
# Pagel's lambda multiplies the off-diagonal elements of the phylogenetic
# covariance and leaves its diagonal alone. The shorebird tree is ultrametric
# and C is normalised to unit root height (diag C = 1 to within 3.7e-8), so the
# transform is lambda * C + (1 - lambda) * I, which is the form used here.
#
# Difference from the fixed-lambda row. sicm/pgls.jl fixes lambda = 1, so its
# covariance sigma^2 C is a scalar times data (TriSolveScale cascade). Here
# lambda is sampled, so no scale rule applies, but the covariance written out
# term by term,
#
#   Sigma = s * C + t * I,   s = sigma^2 lambda,   t = sigma^2 (1 - lambda),
#
# is the CholeskyEigenLift shape. The density keeps the whitened-Cholesky form
# of sicm/pgls.jl (whiten y and X separately against chol(Sigma), take an OLS
# residual in the whitened space), and EigenLift rewrites every user of the
# factor:
#   L \ y, L \ X    ->  Q diag(1/sqrt(s lambda_i + t)) Q' y, same for X,
#                       with Q' y and Q' X sample-invariant and hoisted;
#   diag(L)         ->  sqrt(s lambda_i + t)                 (log-determinant),
# so the per-step O(n^3) Cholesky and O(n^2 (k + 1)) solves become one hoisted
# eigendecomposition of C plus O(n^2 (k + 1)) matrix-vector work per step. The
# whitening is by the symmetric inverse square root instead of by L^{-1}, which
# changes the whitened vectors but not r'r, the only quantity the density uses.
#
# Spelling note: the factored spelling sigma^2 .* (lambda .* C .+ (1 - lambda) .* I)
# is NOT matched by the current rules (CholScale needs the unscaled factor to be
# sample-invariant; EigenLift needs the Cholesky operand to be the sum itself).
# See the evidence dir estimated_parameter_rows_2026-09-24.
#
# Data: caper `shorebird` (71 species; Lislevand & Thomas 2006), same file as
# sicm/pgls.jl. See pplbench/pplbench/models/real_data/README.md.

function _pgls_lambda_sampler(rng, beta, sigma, lambda, C, X, y, n, k)
    return Reactant.Ops.fill(0.0, n)
end

function _pgls_lambda_logpdf(_obs, beta, sigma, lambda, C, X, y, n, k)
    s2d = reshape(sigma, 1, 1)
    l2d = reshape(lambda, 1, 1)
    s2 = s2d .* s2d

    # sigma^2 V(lambda) = (sigma^2 lambda) C + (sigma^2 (1 - lambda)) I
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    Sigma = (s2 .* l2d) .* C .+ (s2 .* (1.0 .- l2d)) .* I_n
    L = Reactant.Ops.cholesky(Sigma; lower=true)      # Sigma = L * L'

    # Whitening solves; both right-hand sides are data.
    Lo = LinearAlgebra.LowerTriangular(L)
    ytilde = Lo \ reshape(y, n, 1)                    # n x 1
    Xtilde = Lo \ X                                   # n x k

    r = ytilde .- Xtilde * reshape(beta, k, 1)
    quad = sum(r .* r)

    # 0.5 * log det(Sigma) = sum(log(diag(L)))
    # diag(L) by masking with the identity and contracting the ROW axis with
    # ones: the spelling Impulse's own emitted MVN density uses (Enzyme
    # DistBuilder, MVN case), which the HLO optimizer turns into
    # dot_general(L, I, batching [1]x[1], contracting [0]x[0]), the diagonal
    # form CholeskyEigenLift recognises. Two equivalent spellings are NOT
    # recognised: `LinearAlgebra.diag(L)` (traces to transpose + flatten +
    # strided slice) and `(L .* I) * ones(n)` (contracts the column axis, giving
    # batching [0]x[0], contracting [1]x[1]); see the evidence dir.
    diagL = transpose(L .* I_n) * ones(n)
    half_logdet = sum(log.(diagL))

    return -0.5 * quad - half_logdet - 0.5 * n * log(2 * pi)
end

# lambda ~ Uniform(0, 1); `support=:unit_interval` makes HMC sample
# logit(lambda) with the sigmoid Jacobian (NumPyro's dist.Uniform(0, 1)).
function _unit_uniform_sampler(rng)
    return rand(rng, 1)
end

function _unit_uniform_logpdf(lambda)
    return sum(0.0 .* lambda)
end

function model_pgls_lambda(rng, C, X, y, n, k)
    _, beta = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (k,)); symbol=:beta)
    _, sigma = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma)
    _, lambda = ProbProg.sample(
        rng,
        _unit_uniform_sampler;
        symbol=:lambda,
        logpdf=_unit_uniform_logpdf,
        support=:unit_interval,
    )
    _, _ = ProbProg.sample(
        rng,
        _pgls_lambda_sampler,
        beta,
        sigma,
        lambda,
        C,
        X,
        y,
        n,
        k;
        symbol=:y,
        logpdf=_pgls_lambda_logpdf,
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

    model_args = (Reactant.to_rarray(C), Reactant.to_rarray(X), Reactant.to_rarray(y), n, k)

    selection = ProbProg.select(
        ProbProg.Address(:beta), ProbProg.Address(:sigma), ProbProg.Address(:lambda)
    )

    return (
        model_fn=model_pgls_lambda,
        model_args=model_args,
        selection=selection,
        position_size=k + 2,
        model_name="PGLS, Pagel's lambda estimated (shorebird)",
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
