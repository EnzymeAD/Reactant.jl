# Phylogenetic Poisson regression (motivating example), COVARIANCE-FORM SPELLING (2026-10-07).
#
# The same model as motivating/phylo_pois.jl (the brms vignette model with brms 2.23.0 default
# priors; data, priors, parameters and selection identical), with ONE difference in how the
# phylogenetic effect is written: the covariance sd_phylo^2 * A is assembled and factorized,
#   r_phylo = chol(sd_phylo^2 * A) * z_phylo,
# which is the model's own definition r_phylo ~ N(0, sd_phylo^2 A) in non-centered form, instead
# of brms's Stan spelling r_phylo = sd_phylo * (Lcov * z_phylo) with Lcov = t(chol(A)) computed in
# R. For sd_phylo > 0 the two are the same function (chol(s^2 A) = s chol(A)), so the log density
# is identical at every point. Variant for the user's decision on the section 2 row; see
# evidence/phylo_model_fix_2026-10-07/RESULTS.md.

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

    # Phylogenetic effect: r_phylo = chol(sd_phylo^2 * A) * z_phylo
    C = cholesky(reshape(sd_phylo .^ 2, 1, 1) .* A)
    r_phylo = reshape(C.factors' * reshape(z_phylo, n, 1), n)
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
