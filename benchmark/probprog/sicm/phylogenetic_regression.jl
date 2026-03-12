# Phylogenetic Regression with Two Covariance Sources (Evolutionary Biology)
#
# Model:
#   y ~ MVN(X * beta, sigma_p^2 * C_phylo + sigma_e^2 * C_env)
#   beta ~ Normal(0, 10)^k          (regression coefficients)
#   sigma_p ~ HalfNormal(1)          (phylogenetic signal std)
#   sigma_e ~ HalfNormal(1)          (environmental signal std)
#   C_phylo = phylogenetic covariance from known tree (sample-invariant)
#   C_env = environmental similarity matrix (sample-invariant, != I)
#
# SICM: **GeneralizedEigenLift** (NEW PATTERN) on sigma_p^2 * C_phylo + sigma_e^2 * C_env
#   Standard EigenLift requires s*A + t*I (identity on the right).
#   GeneralizedEigenLift handles s*A + t*B where BOTH A, B are invariant PD:
#     1. Compute L_B = chol(C_env)                           # invariant, hoisted
#     2. Compute C_w = L_B^{-1} * C_phylo * L_B^{-T}        # whitened, invariant, hoisted
#     3. Compute Q, lambda = eigh(C_w)                       # invariant, hoisted
#     4. Per-iteration: d = sigma_p^2 * lambda + sigma_e^2   # O(N)
#     5. Downstream: Q @ diag(1/d) @ Q^T operations          # O(N^2)
#   Reduces O(N^3) per iteration to O(N^2).
#
# Domain: Comparative biology, eco-evolutionary dynamics.
#   Separating phylogenetic from environmental signal is a central question in
#   comparative methods (Freckleton & Jetz 2009, Ives & Helmus 2011).
#
#   C_phylo[i,j] = shared branch length from a known phylogenetic tree,
#     constructed from molecular sequence data (fixed for a given clade).
#   C_env[i,j] = environmental similarity (climate, geography, niche overlap),
#     computed from species' habitat characteristics (fixed for a given dataset).
#
#   The key quantity of interest is phylogenetic heritability:
#     H = sigma_p^2 / (sigma_p^2 + sigma_e^2)
#   which measures how much trait variation is explained by shared ancestry
#   versus shared environment. This "space versus phylogeny" decomposition
#   appears in community ecology (Helmus et al. 2007), trait evolution
#   (Freckleton & Jetz 2009), and disease ecology (Streicker et al. 2010).
#
#   Both C_phylo and C_env are sample-invariant because they are derived from
#   fixed data (the phylogenetic tree and environmental measurements).
#   Neither is the identity matrix, motivating GeneralizedEigenLift.
#
# Also applies to: space-time models (C_spatial + C_temporal),
# multi-kernel GP regression (K_se + K_matern),
# and social-genetic models (C_kinship + C_social_network).

function model_phylo_regression(rng, X, C_phylo, C_env, y)
    n = Base.length(y)
    k = size(X, 2)

    _, beta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 10.0, (k,)); symbol=:beta
    )

    _, sigma_p = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_p
    )

    _, sigma_e = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_e
    )

    # Mean: X * beta
    mu = reshape(X * reshape(beta, k, 1), n)

    # Covariance: sigma_p^2 * C_phylo + sigma_e^2 * C_env
    # Both C_phylo and C_env are invariant, neither is identity.
    # --> GeneralizedEigenLift (new pattern)
    sp_2d = reshape(sigma_p, 1, 1)
    se_2d = reshape(sigma_e, 1, 1)
    K = (sp_2d .* sp_2d) .* C_phylo .+ (se_2d .* se_2d) .* C_env

    _, obs = ProbProg.sample(
        rng, ProbProg.MultiNormal(mu, K, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:k]
    C_phylo_raw = data["C_phylo"]
    C_phylo = [Float64(C_phylo_raw[i][j]) for i in 1:n, j in 1:n]
    C_env_raw = data["C_env"]
    C_env = [Float64(C_env_raw[i][j]) for i in 1:n, j in 1:n]
    y = Float64.(data["Y"])

    X_rarray = Reactant.to_rarray(X)
    C_phylo_rarray = Reactant.to_rarray(C_phylo)
    C_env_rarray = Reactant.to_rarray(C_env)
    model_args = (X_rarray, C_phylo_rarray, C_env_rarray, y)

    selection = ProbProg.select(
        ProbProg.Address(:beta),
        ProbProg.Address(:sigma_p),
        ProbProg.Address(:sigma_e),
    )

    return (
        model_fn = model_phylo_regression,
        model_args = model_args,
        selection = selection,
        position_size = k + 2,  # beta(k) + sigma_p(1) + sigma_e(1)
        model_name = "Phylogenetic Regression",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    k = Int(attrs["k"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_beta = Float64.(init_params["beta"])
        init_sigma_p = Float64.(init_params["sigma_p"])
        init_sigma_e = Float64.(init_params["sigma_e"])
    else
        init_beta = zeros(k)
        init_sigma_p = [1.0]
        init_sigma_e = [1.0]
    end

    return ProbProg.Constraint(
        :beta => init_beta,
        :sigma_p => init_sigma_p,
        :sigma_e => init_sigma_e,
        :y => y,
    )
end

function extract_samples(trace)
    beta_samples = trace.choices[:beta]
    return Dict{String,Any}(
        "beta" => [collect(beta_samples[i, :]) for i in 1:size(beta_samples, 1)],
        "sigma_p" => collect(vec(trace.choices[:sigma_p])),
        "sigma_e" => collect(vec(trace.choices[:sigma_e])),
    )
end
