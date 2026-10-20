# CAR data model for survey estimates with standard errors, county ICE  -- SICM: EigenLift
# (rho estimated, n = 2875)
#
# Model: Donegan, Chun & Griffith (2021), "Modeling community health with areal data:
# Bayesian inference with survey standard errors and spatial structure", IJERPH 18(13):6856,
# doi:10.3390/ijerph18136856; the data model for the county Index of Concentration at the
# Extremes (eq. 5 with the parameter model of eq. 9):
#   z   ~ Gauss(x, z_se)                          ACS estimates with their standard errors
#   x   ~ MVGauss(mu 1, (I - rho C)^{-1} M)       WCAR: C = D^{-1} A, M = tau^2 D^{-1}
#   mu  ~ Gauss(0, 0.5)
#   tau ~ Gauss+(0, 1)
#   rho ~ Uniform(1 / lambda_min, 1 / lambda_max),  lambda = eig(M^{-1/2} C M^{1/2})
# A is the authors' binary county adjacency (queen contiguity plus their manual
# connections), D = diag(rowsum A), z the mean-centred ICE (survey-HBM README). The
# authors' repository (github.com/ConnorDonegan/survey-HBM, stan/CAR-HBM.stan) gives the
# same model with x_mu ~ std_normal(); we take the paper's eq. 9 prior N(0, 0.5).
#
# Spelling. The CAR density is the authors' own car_normal_lpdf (survey-HBM
# stan/CAR-functions.stan) with ONE change: its log-determinant. Their function takes
#   0.5 log det P = 0.5 sum(log1m(rho lambda)) - n log(tau) (+ a constant dropped there),
# with lambda = eig(M^{-1/2} C M^{1/2}) precomputed once in transformed data (eMCM), a
# hand derivation. Here the log-determinant is spelled as a direct implementation would
# take it, from a Cholesky factor of the precision matrix:
#   P = M^{-1} (I - rho C) = (D - rho A) / tau^2
#     = (-rho / tau^2) * A + (1 / tau^2) * D      written term by term (s A + t B)
#   0.5 log det P = sum(log(diag(chol(P))))
# and the quadratic form is theirs, transcribed (dense C for their CSR (I - rho C)):
#   (x - mu)' P (x - mu) = tau^-2 * dot((x - mu) .* M_inv, (I - rho C)(x - mu)),  M_inv = Ni
# CholeskyEigenLift derives their eigenvalue log-determinant from the Cholesky spelling:
# with B = D (invariant) it whitens through chol(D), eigendecomposes D^{-1/2} A D^{-1/2}
# (= M^{-1/2} C M^{1/2}, their lambda) once, hoisted, and replaces diag(chol(P)) by
# diag(chol(D)) .* sqrt(s lambda + t), so the O(n^3) Cholesky (and its adjoint) of every
# leapfrog step becomes one setup eigendecomposition plus O(n) work. Both configurations keep
# the two dense n x n matrix-vector products of the quadratic form and its gradient.
#
# Term order matters for the matcher (SampleInvariantCodeMotion.cpp,
# matchScaledInvariantPlusDiagSum): the generalized branch whitens through chol of the
# matrix in the second summand's position, so the sum is written A-term first, D-term
# second.
#
# Data: ablation/sicm_roster/data/car_ice_counties_female_n2875.json (the paper's female
# sample, the README's default).

# P(tau, rho) = (-rho / tau^2) A + (1 / tau^2) D
function _wcar_precision(tau, rho, A, D)
    t2d = reshape(tau, 1, 1)
    r2d = reshape(rho, 1, 1)
    inv_tau2 = 1.0 ./ (t2d .* t2d)
    return (-(r2d) .* inv_tau2) .* A .+ inv_tau2 .* D
end

# Exact draw from MVN(mu 1, P^{-1}): with P = L L', x = mu + L^{-T} e has covariance P^{-1}.
# Used by simulate; the MCMC path scores x through the logpdf below.
function _wcar_x_sampler(rng, mu, tau, rho, A, D, C, M_inv, n)
    P = _wcar_precision(tau, rho, A, D)
    L = Reactant.Ops.cholesky(P; lower=true)
    e = randn(rng, n)
    w = Reactant.Ops.triangular_solve(
        L, e; left_side=true, lower=true, transpose_a='T', unit_diagonal=false
    )
    return w .+ reshape(mu, 1)
end

# log N(x | mu 1, P^{-1}) = 0.5 log det P - 0.5 (x - mu)' P (x - mu) - (n/2) log(2 pi)
function _wcar_x_logpdf(x, mu, tau, rho, A, D, C, M_inv, n)
    # quadratic form as car_normal_lpdf computes it
    t2d = reshape(tau, 1)
    r2d = reshape(rho, 1)
    zc = x .- reshape(mu, 1)                          # z = y - mu
    zMinv = zc .* M_inv                               # z' * M^-1
    ImrhoCz = zc .- r2d .* vec(C * reshape(zc, n, 1)) # (I - rho C) z
    quad = sum(zMinv .* ImrhoCz) ./ (t2d .* t2d)

    # 0.5 log det P = sum(log(diag(L)))  ->  CholeskyEigenLift (diagonal extraction)
    P = _wcar_precision(tau, rho, A, D)
    L = Reactant.Ops.cholesky(P; lower=true)
    # diag(L) by masking with the identity and contracting the row axis with ones: the
    # spelling Impulse's own emitted MVN density uses (sicm/car_leroux.jl).
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    diagL = transpose(L .* I_n) * ones(n)
    half_logdet = sum(log.(diagL))

    return half_logdet - 0.5 * sum(quad) - 0.5 * n * log(2 * pi)
end

# rho ~ Uniform(lo, hi); the site carries support=:interval with these bounds, so HMC
# samples the logit-scaled rho and adds the Jacobian (Stan's real<lower=lo, upper=hi>).
function _rho_uniform_sampler(rng, lo, hi)
    return lo .+ (hi - lo) .* rand(rng, 1)
end

function _rho_uniform_logpdf(rho, lo, hi)
    return sum(0.0 .* rho) - log(hi - lo)
end

function model_car_ice(rng, A, D, C, M_inv, z_se, n, rho_lo, rho_hi)
    _, mu = ProbProg.sample(rng, ProbProg.Normal(0.0, 0.5, (1,)); symbol=:mu)
    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    _, rho = ProbProg.sample(
        rng, _rho_uniform_sampler, rho_lo, rho_hi;
        symbol=:rho, logpdf=_rho_uniform_logpdf,
        support=:interval, bounds=(rho_lo, rho_hi),
    )
    # process model: WCAR prior on the true ICE values; density `_wcar_x_logpdf`
    _, x = ProbProg.sample(
        rng, _wcar_x_sampler, mu, tau, rho, A, D, C, M_inv, n;
        symbol=:x, logpdf=_wcar_x_logpdf,
    )
    # data model: survey estimates with their standard errors
    _, z = ProbProg.sample(rng, ProbProg.Normal(x, z_se, (n,)); symbol=:z)
    return z
end

function _ice_adjacency(data, n)
    A = zeros(Float64, n, n)
    for e in data["W_edges"]
        i, j = Int(e[1]), Int(e[2])
        A[i, j] = 1.0
        A[j, i] = 1.0
    end
    return A
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    A = _ice_adjacency(data, n)
    deg = vec(sum(A; dims=2))
    @assert deg == Float64.(data["degree"])
    D = Matrix{Float64}(LinearAlgebra.Diagonal(deg))
    C = A ./ deg                                      # prep_car_data: C <- A / Ni
    M_inv = 1.0 ./ (1.0 ./ deg)                       # M_diag <- 1 / Ni; M_inv = 1 ./ M_diag
    z_se = Float64.(data["z_se"])
    rho_lo = Float64(attrs["rho_lower"])
    rho_hi = Float64(attrs["rho_upper"])

    model_args = (
        Reactant.to_rarray(A), Reactant.to_rarray(D), Reactant.to_rarray(C),
        Reactant.to_rarray(M_inv), Reactant.to_rarray(z_se), n, rho_lo, rho_hi,
    )

    selection = ProbProg.select(
        ProbProg.Address(:mu),
        ProbProg.Address(:tau),
        ProbProg.Address(:rho),
        ProbProg.Address(:x),
    )

    return (
        model_fn=model_car_ice,
        model_args=model_args,
        selection=selection,
        position_size=3 + n,  # mu(1) + tau(1) + rho(1) + x(n)
        model_name="WCAR survey data model, county ICE (rho estimated)",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    z = Float64.(data["z"])

    if init_params !== nothing
        init_mu = Float64.(init_params["mu"])
        init_tau = Float64.(init_params["tau"])
        init_rho = Float64.(init_params["rho"])
        init_x = Float64.(init_params["x"])
    else
        init_mu = [0.0]
        init_tau = [1.0]
        init_rho = [0.5]
        init_x = zeros(n)
    end

    return ProbProg.Constraint(
        :mu => init_mu,
        :tau => init_tau,
        :rho => init_rho,
        :x => init_x,
        :z => z,
    )
end

function extract_samples(trace)
    x_samples = trace.choices[:x]
    return Dict{String,Any}(
        "mu" => collect(vec(trace.choices[:mu])),
        "tau" => collect(vec(trace.choices[:tau])),
        "rho" => collect(vec(trace.choices[:rho])),
        "x" => [collect(x_samples[i, :]) for i in 1:size(x_samples, 1)],
    )
end
