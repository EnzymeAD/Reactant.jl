# Leroux CAR with the spatial-dependence parameter rho ESTIMATED  -- SICM: EigenLift
#
# Model (Leroux, Lei & Breslow 2000), as CARBayes fits it by default
# (Lee 2013, JSS 55(13), `S.CARleroux` with `rho = NULL`):
#   y_i   ~ Poisson(E_i * exp(alpha + phi_i))
#   phi   ~ MVN(0, P^{-1})                  (precision form, Stan's multi_normal_prec)
#   P     = tau * Q(rho),   Q(rho) = rho * (D - W) + (1 - rho) * I
#   tau   = 1 / sigma_phi^2
#   D     = diag(rowsum W), W the 0/1 adjacency of the areal units (DATA)
#   alpha     ~ Normal(0, 10)
#   sigma_phi ~ HalfNormal(1)
#   rho       ~ Uniform(0, 1)
#
# Priors. alpha and sigma_phi are the weakly informative priors of the fixed-rho
# row (sicm/car_spatial.jl); rho ~ Uniform(0, 1) is CARBayes's own prior. We
# deviate from CARBayes's diffuse defaults for the other two (beta ~ N(0, 1e5),
# tau2 ~ InvGamma(1, 0.01)), which are chosen for its Gibbs / Metropolis
# updates; under NUTS such near-improper priors only lengthen trajectories.
#
# Difference from the fixed-rho row. sicm/car_spatial.jl fixes rho = 0.99 so its
# covariance sigma_phi^2 * Q^{-1} is a scalar times data (CholScale). Here rho is
# sampled, so the precision P(sigma_phi, rho) is not a scalar multiple of any
# data matrix and no scale rule applies. It IS the EigenLift shape:
#
#   P = s * R + t * I,   R = D - W (data),   s = tau * rho,   t = tau * (1 - rho)
#
# which is the Leroux precision written out term by term. The density needs
#   phi' P phi                       a matvec, no factorization
#   0.5 * log det P = sum(log(diag(chol(P))))
# and CholeskyEigenLift rewrites the diagonal extraction of chol(s R + t I) into
# sqrt(s * lambda + t) with lambda the eigenvalues of R, so the O(n^3) Cholesky
# per leapfrog step becomes one hoisted eigendecomposition of R plus O(n) work.
#
# Spelling note, recorded rather than hidden: the factored spelling
# tau .* (rho .* R .+ (1 - rho) .* I) is NOT matched by the current rules. The
# CholScale rule needs the unscaled factor to be sample-invariant (here it
# depends on rho) and EigenLift needs the Cholesky operand to be the sum itself,
# not a scalar times the sum. The term-by-term spelling below is the same
# matrix; see the evidence dir estimated_parameter_rows_2026-09-24 for the IR.
#
# Data: Scotland lip cancer, 56 districts (Clayton & Kaldor 1987), or the NYC
# traffic-accident whole map, 1921 census tracts (posteriordb
# `traffic_accident_nyc`). See pplbench/pplbench/models/real_data/README.md and
# pplbench/pplbench/models/car_leroux.py.

# P(sigma_phi, rho) = (tau rho) R + (tau (1 - rho)) I, tau = 1 / sigma_phi^2.
# I_n is built here, not passed in, so tracing folds it into a dense constant
# (same as sicm/linear_mixed_model.jl and sicm/sar_lag.jl).
function _leroux_precision(sigma_phi, rho, R, n)
    sp = reshape(sigma_phi, 1, 1)
    r2d = reshape(rho, 1, 1)
    tau = 1.0 ./ (sp .* sp)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    return (tau .* r2d) .* R .+ (tau .* (1.0 .- r2d)) .* I_n
end

# Exact draw from MVN(0, P^{-1}): with P = L L', phi = L^{-T} z has covariance
# L^{-T} L^{-1} = P^{-1}. Used by simulate; the MCMC path scores phi through the
# logpdf below.
function _leroux_phi_sampler(rng, sigma_phi, rho, R, n)
    P = _leroux_precision(sigma_phi, rho, R, n)
    L = Reactant.Ops.cholesky(P; lower=true)
    z = randn(rng, n)
    return Reactant.Ops.triangular_solve(
        L, z; left_side=true, lower=true, transpose_a='T', unit_diagonal=false
    )
end

# log N(phi | 0, P^{-1}) = 0.5 log det P - 0.5 phi' P phi - (n/2) log(2 pi)
function _leroux_phi_logpdf(phi, sigma_phi, rho, R, n)
    P = _leroux_precision(sigma_phi, rho, R, n)
    phi_col = reshape(phi, n, 1)
    quad = sum(phi_col .* (P * phi_col))

    # 0.5 * log det P = sum(log(diag(L)))  ->  CholeskyEigenLift (diagonal)
    L = Reactant.Ops.cholesky(P; lower=true)
    # diag(L) by masking with the identity and contracting the ROW axis with
    # ones: the spelling Impulse's own emitted MVN density uses (Enzyme
    # DistBuilder, MVN case), which the HLO optimizer turns into
    # dot_general(L, I, batching [1]x[1], contracting [0]x[0]), the diagonal
    # form CholeskyEigenLift recognises. Two equivalent spellings are NOT
    # recognised: `LinearAlgebra.diag(L)` (traces to transpose + flatten +
    # strided slice) and `(L .* I) * ones(n)` (contracts the column axis, giving
    # batching [0]x[0], contracting [1]x[1]); see the evidence dir.
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    diagL = transpose(L .* I_n) * ones(n)
    half_logdet = sum(log.(diagL))

    return half_logdet - 0.5 * quad - 0.5 * n * log(2 * pi)
end

# rho ~ Uniform(0, 1): constant density on the unit interval. The site carries
# `support=:unit_interval`, so HMC samples logit(rho) and adds the sigmoid
# Jacobian, which is exactly NumPyro's treatment of dist.Uniform(0, 1).
function _unit_uniform_sampler(rng)
    return rand(rng, 1)
end

function _unit_uniform_logpdf(rho)
    return sum(0.0 .* rho)
end

function model_car_leroux(rng, R, log_E, y, n)
    _, alpha = ProbProg.sample(rng, ProbProg.Normal(0.0, 10.0, (1,)); symbol=:alpha)

    _, sigma_phi = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_phi)

    _, rho = ProbProg.sample(
        rng,
        _unit_uniform_sampler;
        symbol=:rho,
        logpdf=_unit_uniform_logpdf,
        support=:unit_interval,
    )

    # Leroux CAR prior in precision form; the density is `_leroux_phi_logpdf`.
    _, phi = ProbProg.sample(
        rng,
        _leroux_phi_sampler,
        sigma_phi,
        rho,
        R,
        n;
        symbol=:phi,
        logpdf=_leroux_phi_logpdf,
    )

    # Poisson likelihood: y_i ~ Poisson(E_i * exp(alpha + phi_i))
    log_rate = log_E .+ reshape(alpha, 1) .+ phi
    _, obs = ProbProg.sample(rng, ProbProg.Poisson(exp.(log_rate), (n,)); symbol=:y)
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    log_E = Float64.(data["log_E"])
    y = Float64.(data["Y"])

    model_args = (Reactant.to_rarray(R), Reactant.to_rarray(log_E), y, n)

    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:sigma_phi),
        ProbProg.Address(:rho),
        ProbProg.Address(:phi),
    )

    return (
        model_fn=model_car_leroux,
        model_args=model_args,
        selection=selection,
        position_size=3 + n,  # alpha(1) + sigma_phi(1) + rho(1) + phi(n)
        model_name="Leroux CAR, rho estimated",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_sigma_phi = Float64.(init_params["sigma_phi"])
        init_rho = Float64.(init_params["rho"])
        init_phi = Float64.(init_params["phi"])
    else
        init_alpha = [0.0]
        init_sigma_phi = [1.0]
        init_rho = [0.5]
        init_phi = zeros(n)
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :sigma_phi => init_sigma_phi,
        :rho => init_rho,
        :phi => init_phi,
        :y => y,
    )
end

function extract_samples(trace)
    phi_samples = trace.choices[:phi]
    return Dict{String,Any}(
        "alpha" => collect(vec(trace.choices[:alpha])),
        "sigma_phi" => collect(vec(trace.choices[:sigma_phi])),
        "rho" => collect(vec(trace.choices[:rho])),
        "phi" => [collect(phi_samples[i, :]) for i in 1:size(phi_samples, 1)],
    )
end
