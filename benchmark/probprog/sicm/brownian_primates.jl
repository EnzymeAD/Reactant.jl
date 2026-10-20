# Brownian-motion phylogenetic regression of primate brain size  -- SICM: CholeskyScale,
# TriangularSolveScale (x2), DotGeneralScale
#
# Model: McElreath, Statistical Rethinking, 2nd ed. (2020), section 14.5.2, R code
# 14.47-14.51, model m14.10 ("Brownian motion model"):
#   B ~ multi_normal(mu, SIGMA)
#   mu    = a + bM * M + bG * G
#   SIGMA = R * sigma_sq
#   a     ~ normal(0, 1)
#   c(bM, bG) ~ normal(0, 0.5)
#   sigma_sq ~ exponential(1)
# R = vcv(corBrownian(tree)) / max(vcv), the Brownian-motion correlation of the primate
# phylogeny (10kTrees consensus, rethinking's Primates301_nex) trimmed to the 151 species
# with complete brain, body mass and group size records; B, M, G are the standardized logs
# of brain volume, body mass and group size (rethinking::standardize). The source fixes the
# covariance shape: a published Brownian-motion PGLS, where only the scale sigma_sq is
# estimated, so R is DATA and sample-invariant.
#
# Spelling. The density is the whitened-Cholesky form of GLS, the same spelling as
# sicm/pgls.jl (the shorebird row this one replaces): factor SIGMA, whiten the response and
# the design separately, then an ordinary least-squares residual in the whitened space.
# ulam compiles m14.10 to Stan's multi_normal(mu, SIGMA), which factors SIGMA every
# evaluation; the whitened spelling is the numerically preferred form of the same density.
#
#   L      = chol(sigma_sq * R)
#   Btilde = L \ B            n x 1 solve, right-hand side is data
#   Xtilde = L \ X            n x 3 solve, X = [1, M, G], right-hand side is data
#   r      = Btilde - Xtilde * [a, bM, bG]
#   log p  = -r'r / 2 - sum(log diag L) - (n/2) log(2 pi)
#
# SICM cascade (as in sicm/pgls.jl):
#   CholeskyScaleFactorization:        chol(sigma_sq R)         -> sqrt(sigma_sq) chol(R)
#   TriangularSolveScaleFactorization: solve(s L_R, B), solve(s L_R, X)
#                                                               -> (1/s) solve(L_R, .)
#   DotGeneralScaleFactorization:      ((1/s) solve(L_R, X)) beta -> (1/s) (solve(L_R, X) beta)
# so chol(R), solve(L_R, B) and solve(L_R, X) are sample-invariant and leave the leapfrog
# step; the step keeps an n x 3 matrix-vector product and O(n) work.
#
# Data: ablation/sicm_roster/data/brownian_primates_n151.json (n = 151, k = 3).

# Placeholder sampler for the observed site: the value is supplied by the constraint and
# the density enters through `_brownian_logpdf` (convention of sicm/pgls.jl).
function _brownian_sampler(rng, beta, sigma_sq, R, X, y, n, k)
    return Reactant.Ops.fill(0.0, n)
end

function _brownian_logpdf(_obs, beta, sigma_sq, R, X, y, n, k)
    s2d = reshape(sigma_sq, 1, 1)

    # SIGMA <- R * sigma_sq  ->  CholeskyScaleFactorization
    Sigma = R .* s2d
    L = Reactant.Ops.cholesky(Sigma; lower=true)      # SIGMA = L * L'

    # Whitening solves; both right-hand sides are data.
    Lo = LinearAlgebra.LowerTriangular(L)
    ytilde = Lo \ reshape(y, n, 1)                    # n x 1
    Xtilde = Lo \ X                                   # n x k

    r = ytilde .- Xtilde * reshape(beta, k, 1)
    quad = sum(r .* r)

    # 0.5 * log det(SIGMA) = sum(log(diag(L)))
    half_logdet = sum(log.(LinearAlgebra.diag(L)))

    return -0.5 * quad - half_logdet - 0.5 * n * log(2 * pi)
end

function model_brownian_primates(rng, R, X, y, n, k)
    _, a = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:a)
    _, b = ProbProg.sample(rng, ProbProg.Normal(0.0, 0.5, (2,)); symbol=:b)
    _, sigma_sq = ProbProg.sample(rng, ProbProg.Exponential(1.0, (1,)); symbol=:sigma_sq)
    beta = vcat(a, b)                                 # (a, bM, bG), the columns of X
    _, _ = ProbProg.sample(
        rng, _brownian_sampler, beta, sigma_sq, R, X, y, n, k;
        symbol=:y, logpdf=_brownian_logpdf,
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    y = Float64.(data["Y"])

    model_args = (
        Reactant.to_rarray(R),
        Reactant.to_rarray(X),
        Reactant.to_rarray(y),
        n,
        k,
    )

    selection = ProbProg.select(
        ProbProg.Address(:a),
        ProbProg.Address(:b),
        ProbProg.Address(:sigma_sq),
    )

    return (
        model_fn=model_brownian_primates,
        model_args=model_args,
        selection=selection,
        position_size=4,  # a(1) + b(2) + sigma_sq(1)
        model_name="Brownian PGLS (Primates301, m14.10)",
    )
end

function build_constraint(data, init_params)
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_a = Float64.(init_params["a"])
        init_b = Float64.(init_params["b"])
        init_sigma_sq = Float64.(init_params["sigma_sq"])
    else
        init_a = [0.0]
        init_b = zeros(2)
        init_sigma_sq = [1.0]
    end

    return ProbProg.Constraint(
        :a => init_a,
        :b => init_b,
        :sigma_sq => init_sigma_sq,
        :y => y,
    )
end

function extract_samples(trace)
    b_samples = trace.choices[:b]
    return Dict{String,Any}(
        "a" => collect(vec(trace.choices[:a])),
        "b" => [collect(b_samples[i, :]) for i in 1:size(b_samples, 1)],
        "sigma_sq" => collect(vec(trace.choices[:sigma_sq])),
    )
end
