# Phylogenetic Poisson regression (motivating example)
#
# Data: brms phylogenetics vignette count example, simulated by the book authors (200 species
# on a simulated tree), from Modern Phylogenetic Comparative Methods ch. 11 (de Villemereuil &
# Nakagawa 2014).
# Provenance + files: motivating/phylo_data/README.md. Data prep (tree -> A, brms row order,
# raw cofactor): evidence/phylo_model_fix_2026-10-07/scripts/brms_standata.R.
#
# Model: the brms vignette model as written,
#   brm(phen_pois ~ cofactor + (1|gr(phylo, cov = A)) + (1|obs), family = poisson("log"),
#       data2 = list(A = A))
# with the brms 2.23.0 default priors, ported from the Stan code brms generates for it
# (evidence/phylo_model_fix_2026-10-07/stan/brms_phylo_pois.stan; brms names in brackets):
#   b         flat                         population-level slope           [b]
#   Intercept student_t(3, 0.3, 2.6)       intercept of the centered model  [Intercept]
#   sd_phylo  student_t(3, 0, 2.6), > 0    phylogenetic sd                  [sd_1]
#   z_phylo   N(0, 1)^n                                                     [z_1]
#   sd_obs    student_t(3, 0, 2.6), > 0    observation-level sd             [sd_2]
#   z_obs     N(0, 1)^n                                                     [z_2]
#   r_phylo = sd_phylo * (chol(A) * z_phylo)     [r_1_1 = sd_1[1] * (Lcov_1 * z_1[1])]
#   r_obs   = sd_obs * z_obs                     [r_2_1 = sd_2[1] * z_2[1]]
#   k ~ Poisson(exp(Intercept + (x - mean(x)) * b + r_phylo + r_obs))   [poisson_log_glm]
# The prior literals are the data-dependent brms defaults (priors.R def_scale_prior: location
# round(median(log y'), 1) = 0.3 and scale max(2.5, round(mad(log y'), 1)) = 2.6, where y' is
# y with zeros replaced by 0.1). x is the RAW cofactor; brms centers it in transformed data and
# reports b_Intercept = Intercept - mean(x) * b. brms computes Lcov = t(chol(A)) once in R
# before Stan runs; here A is the model input and its Cholesky is written where the model
# puts it, so hoisting it out of the sampler is the compiler's job. A is ape::vcv.phylo (a
# DATA input): the rows of A, x and k must be in the same species order.
# Sampler setting (setup below): a diagonal inverse mass matrix, Stan's and brms's default
# metric (diag_e).

const PHYLO_INTERCEPT_LOC = 0.3
const PHYLO_PRIOR_SCALE = 2.6

# Flat prior on the slope b (brms default for population-level effects): log density 0.
_phylo_flat_sampler(rng) = randn(rng, 1)
_phylo_flat_logpdf(b) = sum(0.0 .* b)

function model_phylo_pois(rng, A, x, k)
    n = Base.length(x)

    _, b = ProbProg.sample(
        rng, _phylo_flat_sampler; symbol=:b, logpdf=_phylo_flat_logpdf
    )
    _, Intercept = ProbProg.sample(
        rng, ProbProg.StudentT(3.0, PHYLO_INTERCEPT_LOC, PHYLO_PRIOR_SCALE, (1,));
        symbol=:Intercept,
    )
    _, sd_phylo = ProbProg.sample(
        rng, ProbProg.HalfStudentT(3.0, PHYLO_PRIOR_SCALE, (1,)); symbol=:sd_phylo
    )
    _, z_phylo = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:z_phylo
    )
    _, sd_obs = ProbProg.sample(
        rng, ProbProg.HalfStudentT(3.0, PHYLO_PRIOR_SCALE, (1,)); symbol=:sd_obs
    )
    _, z_obs = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:z_obs
    )

    # Phylogenetic effect: r_phylo = sd_phylo * (chol(A) * z_phylo)
    C = cholesky(A)
    r_phylo = reshape(sd_phylo, 1) .* reshape(C.factors' * reshape(z_phylo, n, 1), n)
    r_obs = reshape(sd_obs, 1) .* z_obs

    xc = x .- sum(x) ./ n
    mu = reshape(Intercept, 1) .+ xc .* reshape(b, 1) .+ r_phylo .+ r_obs
    _, obs = ProbProg.sample(
        rng, ProbProg.Poisson(exp.(mu), (n,)); symbol=:k
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    haskey(attrs, "int_loc") && @assert Float64(attrs["int_loc"]) == PHYLO_INTERCEPT_LOC
    haskey(attrs, "sd_scale") && @assert Float64(attrs["sd_scale"]) == PHYLO_PRIOR_SCALE

    A_raw = data["A"]
    A = [Float64(A_raw[i][j]) for i in 1:n, j in 1:n]
    x = Float64.(data["X"])
    k = Float64.(data["Y"])

    A_rarray = Reactant.to_rarray(A)
    x_rarray = Reactant.to_rarray(x)
    k_rarray = Reactant.to_rarray(k)
    model_args = (A_rarray, x_rarray, k_rarray)

    selection = ProbProg.select(
        ProbProg.Address(:b),
        ProbProg.Address(:Intercept),
        ProbProg.Address(:sd_phylo),
        ProbProg.Address(:z_phylo),
        ProbProg.Address(:sd_obs),
        ProbProg.Address(:z_obs),
    )

    return (
        model_fn = model_phylo_pois,
        model_args = model_args,
        selection = selection,
        position_size = 2 * n + 4,
        model_name = "Phylogenetic Poisson Regression",
        inverse_mass_matrix = ones(Float64, 2 * n + 4),
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    k = Float64.(data["Y"])

    if init_params !== nothing
        init_b = Float64.(init_params["b"])
        init_Intercept = Float64.(init_params["Intercept"])
        init_sd_phylo = Float64.(init_params["sd_phylo"])
        init_z_phylo = Float64.(init_params["z_phylo"])
        init_sd_obs = Float64.(init_params["sd_obs"])
        init_z_obs = Float64.(init_params["z_obs"])
    else
        init_b = [0.0]
        init_Intercept = [0.0]
        init_sd_phylo = [1.0]
        init_z_phylo = zeros(n)
        init_sd_obs = [1.0]
        init_z_obs = zeros(n)
    end

    return ProbProg.Constraint(
        :b => init_b,
        :Intercept => init_Intercept,
        :sd_phylo => init_sd_phylo,
        :z_phylo => init_z_phylo,
        :sd_obs => init_sd_obs,
        :z_obs => init_z_obs,
        :k => k,
    )
end

function extract_samples(trace)
    zp = trace.choices[:z_phylo]
    zo = trace.choices[:z_obs]
    return Dict{String,Any}(
        "b" => collect(vec(trace.choices[:b])),
        "Intercept" => collect(vec(trace.choices[:Intercept])),
        "sd_phylo" => collect(vec(trace.choices[:sd_phylo])),
        "z_phylo" => [collect(zp[i, :]) for i in 1:size(zp, 1)],
        "sd_obs" => collect(vec(trace.choices[:sd_obs])),
        "z_obs" => [collect(zo[i, :]) for i in 1:size(zo, 1)],
    )
end
