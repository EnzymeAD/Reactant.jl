# Phylogenetic Poisson regression (motivating example)
#
# Real data: brms phylogenetics vignette count example (200 species on a known tree),
# from Modern Phylogenetic Comparative Methods ch. 11 (de Villemereuil & Nakagawa 2014).
# Provenance + files: motivating/phylo_data/README.md. Data prep (tree -> A):
# pplbench/pplbench/models/phylo_pois.py.
#
# Model (brms: phen_pois ~ cofactor + (1|gr(phylo, cov = A)) + (1|obs), poisson), written
# in the natural non-centered form. A is the tree covariance (ape::vcv.phylo): a DATA
# input, hence sample-invariant by construction. The only thing inferred about the
# covariance is its scale sigma_sq, sampled as a VARIANCE (Statistical Rethinking m14.10:
# `sigma_sq ~ exponential(1)`, `SIGMA <- R * sigma_sq`), so SICM's
#     cholesky(sigma_sq * A) -> sqrt(sigma_sq) * cholesky(A)
# introduces a genuinely new sqrt and hoists the O(n^3) Cholesky of A.

function model_phylo_pois(rng, A, x, k)
    n = Base.length(x)

    _, sigma_sq = ProbProg.sample(
        rng, ProbProg.Exponential(1.0, (1,)); symbol=:sigma_sq
    )
    _, eta = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:eta
    )
    _, b0 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:b0
    )
    _, b1 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:b1
    )
    _, sigma_obs = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma_obs
    )
    _, eps = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:eps
    )

    # Phylogenetic random effect: f = chol(sigma_sq * A) * eta
    s_2d = reshape(sigma_sq, 1, 1)
    cov = s_2d .* A
    C = cholesky(cov)
    f = reshape(C.factors' * reshape(eta, n, 1), n)

    log_rate = reshape(b0, 1) .+ reshape(b1, 1) .* x .+ f .+ reshape(sigma_obs, 1) .* eps
    _, obs = ProbProg.sample(
        rng, ProbProg.Poisson(exp.(log_rate), (n,)); symbol=:k
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    A_raw = data["A"]
    A = [Float64(A_raw[i][j]) for i in 1:n, j in 1:n]
    x = Float64.(data["X"])
    k = Float64.(data["Y"])

    A_rarray = Reactant.to_rarray(A)
    x_rarray = Reactant.to_rarray(x)
    k_rarray = Reactant.to_rarray(k)
    model_args = (A_rarray, x_rarray, k_rarray)

    selection = ProbProg.select(
        ProbProg.Address(:sigma_sq),
        ProbProg.Address(:eta),
        ProbProg.Address(:b0),
        ProbProg.Address(:b1),
        ProbProg.Address(:sigma_obs),
        ProbProg.Address(:eps),
    )

    return (
        model_fn = model_phylo_pois,
        model_args = model_args,
        selection = selection,
        position_size = 2 * n + 4,
        model_name = "Phylogenetic Poisson Regression",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    k = Float64.(data["Y"])

    if init_params !== nothing
        init_sigma_sq = Float64.(init_params["sigma_sq"])
        init_eta = Float64.(init_params["eta"])
        init_b0 = Float64.(init_params["b0"])
        init_b1 = Float64.(init_params["b1"])
        init_sigma_obs = Float64.(init_params["sigma_obs"])
        init_eps = Float64.(init_params["eps"])
    else
        init_sigma_sq = [1.0]
        init_eta = zeros(n)
        init_b0 = [0.0]
        init_b1 = [0.0]
        init_sigma_obs = [1.0]
        init_eps = zeros(n)
    end

    return ProbProg.Constraint(
        :sigma_sq => init_sigma_sq,
        :eta => init_eta,
        :b0 => init_b0,
        :b1 => init_b1,
        :sigma_obs => init_sigma_obs,
        :eps => init_eps,
        :k => k,
    )
end

function extract_samples(trace)
    eta_samples = trace.choices[:eta]
    eps_samples = trace.choices[:eps]
    return Dict{String,Any}(
        "sigma_sq" => collect(vec(trace.choices[:sigma_sq])),
        "eta" => [collect(eta_samples[i, :]) for i in 1:size(eta_samples, 1)],
        "b0" => collect(vec(trace.choices[:b0])),
        "b1" => collect(vec(trace.choices[:b1])),
        "sigma_obs" => collect(vec(trace.choices[:sigma_obs])),
        "eps" => [collect(eps_samples[i, :]) for i in 1:size(eps_samples, 1)],
    )
end
