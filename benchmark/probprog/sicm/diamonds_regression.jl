# Diamonds log-log price regression (SICM: GaussianSufficientStats)
#
# posteriordb posterior `diamonds-diamonds`, brms 2.10.0. The Stan file
# (posterior_database/models/stan/diamonds.stan) is
#
#   Intercept ~ student_t(3, 8, 10)
#   b         ~ normal(0, 1)                    (Kc = K - 1 = 24 coefficients)
#   sigma     ~ student_t(3, 0, 10) T[0,]       (half student-t)
#   Y         ~ normal_id_glm(Xc, Intercept, b, sigma)
#
# with Xc the K - 1 non-intercept columns of the brms design matrix, centered.
#
# Here the intercept is folded back into the design matrix: with
# X = [1, Xc] (N x K) and beta = [Intercept; b] (K,), the mean is the single
# product X * beta, which is the identical likelihood with the identical
# parameters and priors. The pplbench model class does the centering and hands
# this file the augmented X.
#
# Every prior is the Stan file's. sigma's is ProbProg.HalfStudentT(3, 10), the
# student_t(3, 0, 10) density truncated to sigma > 0, i.e. Stan's
# student_t_lpdf(sigma | 3, 0, 10) - student_t_lccdf(0 | 3, 0, 10) (the
# normalizer is log(0.5)); in NumPyro,
# TruncatedDistribution(StudentT(3, 0, 10), low=0).
#
# SICM GaussianSufficientStats pattern:
#   sum((y - X beta)^2) is answered from the thin QR of the sample-invariant X:
#   X = Q1 R, c = Q1'y, rem = ||y - Q1 c||^2 all hoist, and the in-loop residual
#   becomes the K-vector c - R beta.
#   Per gradient: O(N K) + several O(N) kernels -> O(K^2).
#   Because SICM runs before AD, the reverse pass collapses to K-dim work too.

import Libdl

# Register the LAPACK QR runtime symbols that the GaussianSufficientStats QR
# form needs.
#
# The rule emits `enzymexla.linalg.qr`, which lowers to calls to
# `enzymexla_lapack_dgeqrf_` and `enzymexla_lapack_dorgqr_`. Reactant's
# `TracedLinearAlgebra.__init__` (src/stdlibs/LinearAlgebra.jl) maps gtsv,
# getrf, gesvd, gesdd, gesvj, syevd, syrk, trmm, symm, potrf and trsm, but has
# NO geqrf/orgqr entry, so those two calls do not resolve and the whole
# "impulse" pass pipeline fails at compile time with
#   JIT session error: Symbols not found: [ enzymexla_lapack_dgeqrf_ ]
#
# SUPERSEDED 2026-08-25. The four entries now live permanently in Reactant's
# own symbol table, `TracedLinearAlgebra.__init__` in
# src/stdlibs/LinearAlgebra.jl, in the QR block just after getrf. The call site
# in `setup` below is therefore commented out, and this function is kept only so
# the workaround can be restored in one line if that registration ever regresses.
# Verified end to end by the diamonds smoke on 2026-08-25 with the call disabled.
function register_qr_lapack_symbols!()
    handle = Libdl.dlopen(LinearAlgebra.BLAS.libblas)
    for (cname, enzymexla_name) in [
        (LinearAlgebra.BLAS.@blasfunc(sgeqrf_), :enzymexla_lapack_sgeqrf_),
        (LinearAlgebra.BLAS.@blasfunc(dgeqrf_), :enzymexla_lapack_dgeqrf_),
        (LinearAlgebra.BLAS.@blasfunc(sorgqr_), :enzymexla_lapack_sorgqr_),
        (LinearAlgebra.BLAS.@blasfunc(dorgqr_), :enzymexla_lapack_dorgqr_),
    ]
        Reactant.MLIR.API.EnzymeJaXMapSymbol(
            enzymexla_name, Libdl.dlsym(handle, cname)
        )
    end
    return nothing
end

function model_diamonds(rng, X, N, K)
    Kc = K - 1

    # Stan: target += student_t_lpdf(Intercept | 3, 8, 10)
    _, intercept = ProbProg.sample(
        rng, ProbProg.StudentT(3.0, 8.0, 10.0, (1,)); symbol=:intercept
    )

    # Stan: target += normal_lpdf(b | 0, 1)
    _, b = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (Kc,)); symbol=:b
    )

    # Stan: target += student_t_lpdf(sigma | 3, 0, 10)
    #                 - 1 * student_t_lccdf(0 | 3, 0, 10)
    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfStudentT(3.0, 10.0, (1,)); symbol=:sigma
    )

    # beta = [Intercept; b], so mu = X * beta = Intercept + Xc * b.
    #
    # `X * beta` with a 1-D beta on purpose: it emits a single
    # dot_general(X, beta) whose right operand is rank 1, which is the shape
    # matchGaussianResidual walks up to. The `(X * reshape(beta, K, 1))[:, 1]`
    # spelling used by some of the other sicm models would put a reshape/slice
    # between the residual subtract and the dot and the rule would not fire.
    beta = vcat(intercept, b)
    mu = X * beta

    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(mu, sigma, (N,)); symbol=:y
    )
    return obs
end

function setup(data)
    # Not needed since 2026-08-25: the sgeqrf/dgeqrf/sorgqr/dorgqr entries are
    # registered permanently in Reactant's src/stdlibs/LinearAlgebra.jl.
    # Restore this line if that ever regresses.
    # register_qr_lapack_symbols!()

    attrs = data["attrs"]
    N = Int(attrs["n"])
    K = Int(attrs["k"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:N, j in 1:K]

    X_rarray = Reactant.to_rarray(X)
    model_args = (X_rarray, N, K)

    selection = ProbProg.select(
        ProbProg.Address(:intercept),
        ProbProg.Address(:b),
        ProbProg.Address(:sigma),
    )

    return (
        model_fn = model_diamonds,
        model_args = model_args,
        selection = selection,
        position_size = K + 1,
        model_name = "Diamonds Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    K = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_beta = zeros(K)
        init_beta[1] = 8.0
        init_sigma = [1.0]
    end

    return ProbProg.Constraint(
        :intercept => init_beta[1:1],
        :b => init_beta[2:K],
        :sigma => init_sigma,
        :y => y,
    )
end

function extract_samples(trace)
    intercept = trace.choices[:intercept]
    b_samples = trace.choices[:b]
    ndraws = size(b_samples, 1)
    return Dict{String,Any}(
        "beta" => [
            vcat(Float64(intercept[i, 1]), collect(b_samples[i, :]))
            for i in 1:ndraws
        ],
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
