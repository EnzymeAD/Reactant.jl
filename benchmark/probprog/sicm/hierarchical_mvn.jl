# Hierarchical Multivariate Model with Per-Outcome Scales (Biostatistics)
#
# Model:
#   J groups, each with K correlated outcomes.
#   eta ~ Normal(0, 1)^{J*K}        (standardized group effects)
#   mu ~ Normal(0, 10)^K            (grand mean per outcome)
#   sigma_k ~ HalfNormal(1)^K       (per-outcome between-group std devs)
#   sigma_obs ~ HalfNormal(1)       (observation noise)
#
#   NCP transform:
#     Sigma = diag(sigma) * Omega * diag(sigma)    (K x K covariance)
#     L = chol(Sigma)
#     theta_j = mu + L * eta_j       for each group j
#     y_{jk} ~ Normal(theta_{jk}, sigma_obs)
#
#   Omega = known K x K outcome correlation matrix (sample-invariant).
#
# SICM: **DiagonalScaleCholeskyFactorization** (NEW PATTERN)
#   chol(D * A * D) = D * chol(A)   where D = diag(sigma), A = Omega
#
#   Algebraic identity: if A = L*L', then D*A*D = (D*L)*(D*L)'.
#   Since D is diagonal, D*L is lower triangular with positive diagonal.
#   So chol(D*A*D) = D * chol(A).
#
#   chol(Omega) is invariant -> hoisted (O(K^3) once).
#   Per-iteration: D * L_Omega (O(K^2) diagonal scaling).
#
#   Downstream cascade:
#     log(diag(D*L)) = log(sigma) + log(diag(L))  -> log(diag(L)) hoisted
#     tri_solve(D*L, b) = tri_solve(L, D^{-1}*b)  -> uses hoisted L
#
# Domain: Multivariate meta-analysis, multi-outcome clinical trials.
#   The canonical parameterization for between-study covariance in multivariate
#   meta-analysis (Jackson, Riley & White 2011, Riley et al. 2017):
#     Sigma_between = diag(tau) * R * diag(tau)
#   where tau_k are outcome-specific heterogeneity scales and R is the
#   between-outcome correlation matrix.
#
#   Why Omega (= R) is known / fixed:
#     In network meta-analysis and multivariate treatment comparisons, the
#     correlation structure R is often estimated from a large reference dataset
#     or from domain knowledge (e.g., known correlation between efficacy and
#     safety endpoints from prior trials), then fixed for the analysis of a
#     specific trial or meta-analysis. This is standard practice when K is
#     small (2-10 outcomes) and individual studies lack power to estimate R
#     (Lu & Ades 2009, Wei & Higgins 2013).
#
#   More flexible than single-tau model (scale_family_mvn with tau^2 * Omega):
#     Single tau: all outcomes share the same between-group variance ratio.
#     Per-outcome sigma_k: each outcome has its own variance scale, only
#     correlation is shared. This is the recommended default (Riley 2009).
#
#   Concrete examples:
#     - Multi-endpoint clinical trial: K = efficacy + safety + biomarker (K=3)
#     - Multivariate meta-analysis: K = systolic + diastolic BP (K=2)
#     - Educational testing: K = math + reading + science scores (K=3)
#     - Multi-trait GWAS: K = height + BMI + waist-hip ratio (K=3-5)

function model_hierarchical_mvn(rng, Omega, K, J)
    _, mu = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (K,)); symbol=:mu
    )

    # Per-outcome between-group scales (K independent parameters)
    _, sigma_k = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (K,)); symbol=:sigma_k
    )

    _, sigma_obs = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_obs
    )

    _, eta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (J * K,)); symbol=:eta
    )

    # Covariance: diag(sigma_k) * Omega * diag(sigma_k)
    # = (sigma_k ⊗ sigma_k^T) ⊙ Omega
    # --> DiagonalScaleCholeskyFactorization (new pattern)
    sigma_col = reshape(sigma_k, K, 1)   # K x 1
    sigma_row = reshape(sigma_k, 1, K)   # 1 x K
    cov = (sigma_col .* sigma_row) .* Omega   # element-wise: sigma_i * sigma_j * Omega_ij

    C = cholesky(cov)

    # Batched NCP: theta_j = mu + L * eta_j
    eta_matrix = reshape(eta, K, J)
    mu_col = reshape(mu, K, 1)
    theta = mu_col .+ C.factors' * eta_matrix   # K x J

    # Observations: y_{jk} ~ Normal(theta_{jk}, sigma_obs)
    theta_flat = reshape(theta, J * K)
    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(theta_flat, sigma_obs, (J * K,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    K = Int(attrs["K"])
    J = Int(attrs["J"])

    Omega_raw = data["Omega"]
    Omega = [Float64(Omega_raw[i][j]) for i in 1:K, j in 1:K]

    y_raw = data["Y"]
    y_matrix = [Float64(y_raw[k][j]) for k in 1:K, j in 1:J]

    Omega_rarray = Reactant.to_rarray(Omega)
    model_args = (Omega_rarray, K, J)

    selection = ProbProg.select(
        ProbProg.Address(:mu),
        ProbProg.Address(:sigma_k),
        ProbProg.Address(:sigma_obs),
        ProbProg.Address(:eta),
    )

    return (
        model_fn = model_hierarchical_mvn,
        model_args = model_args,
        selection = selection,
        position_size = 2 * K + 1 + J * K,  # mu(K) + sigma_k(K) + sigma_obs(1) + eta(J*K)
        model_name = "Hierarchical MVN",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    K = Int(attrs["K"])
    J = Int(attrs["J"])

    y_raw = data["Y"]
    y_matrix = [Float64(y_raw[k][j]) for k in 1:K, j in 1:J]
    y_flat = vec(y_matrix)

    if init_params !== nothing
        init_mu = Float64.(init_params["mu"])
        init_sigma_k = Float64.(init_params["sigma_k"])
        init_sigma_obs = Float64.(init_params["sigma_obs"])
        init_eta = Float64.(init_params["eta"])
    else
        init_mu = zeros(K)
        init_sigma_k = ones(K)
        init_sigma_obs = [1.0]
        init_eta = zeros(J * K)
    end

    return ProbProg.Constraint(
        :mu => init_mu,
        :sigma_k => init_sigma_k,
        :sigma_obs => init_sigma_obs,
        :eta => init_eta,
        :y => y_flat,
    )
end

function extract_samples(trace)
    mu_samples = trace.choices[:mu]
    sigma_k_samples = trace.choices[:sigma_k]
    eta_samples = trace.choices[:eta]
    return Dict{String,Any}(
        "mu" => [collect(mu_samples[i, :]) for i in 1:size(mu_samples, 1)],
        "sigma_k" => [collect(sigma_k_samples[i, :]) for i in 1:size(sigma_k_samples, 1)],
        "sigma_obs" => collect(vec(trace.choices[:sigma_obs])),
        "eta" => [collect(eta_samples[i, :]) for i in 1:size(eta_samples, 1)],
    )
end
