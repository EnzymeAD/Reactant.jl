# Brownian-motion phylogenetic regression of primate brain size, written as the book writes it
#
# Model: McElreath, Statistical Rethinking, 2nd ed. (2020), section 14.5.2, R code
# 14.47-14.51, model m14.10 ("Brownian motion model"):
#   B ~ multi_normal(mu, SIGMA)
#   mu    <- a + bM * M + bG * G
#   SIGMA <- R * sigma_sq
#   a     ~ normal(0, 1)
#   c(bM, bG) ~ normal(0, 0.5)
#   sigma_sq ~ exponential(1)
# R = vcv(corBrownian(tree)) / max(vcv), the Brownian-motion correlation of the primate
# phylogeny (10kTrees consensus, rethinking's Primates301_nex) trimmed to the 151 species
# with complete brain, body mass and group size records; B, M, G are the standardized logs
# of brain volume, body mass and group size (rethinking::standardize). R is data.
#
# Spelling. Line for line the book's model: the mean and the covariance are formed as in
# R code 14.51, sigma_sq is sampled as a variance, and the response is scored by the
# front-end MultivariateNormal on (mu, SIGMA), whose density factors SIGMA and solves
# against the factor at every evaluation (the density Stan's multi_normal computes). No
# whitening, no Cholesky factor of R, no reparameterization: the same row as
# sicm/brownian_primates.jl, whose whitened-GLS spelling is an expert rewrite of this one.
#
# Data: ablation/sicm_roster/data/brownian_primates_n151.json (n = 151; X = [1, M, G]).

function model_brownian_primates_natural(rng, M, G, R, n)
    _, a = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:a)
    _, bM = ProbProg.sample(rng, ProbProg.Normal(0.0, 0.5, (1,)); symbol=:bM)
    _, bG = ProbProg.sample(rng, ProbProg.Normal(0.0, 0.5, (1,)); symbol=:bG)
    _, sigma_sq = ProbProg.sample(rng, ProbProg.Exponential(1.0, (1,)); symbol=:sigma_sq)

    mu = a .+ bM .* M .+ bG .* G
    SIGMA = R .* sigma_sq

    _, B = ProbProg.sample(rng, ProbProg.MultivariateNormal(mu, SIGMA, (n,)); symbol=:B)
    return B
end

function setup(data)
    n = Int(data["attrs"]["n"])

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    X_raw = data["X"]
    M = [Float64(X_raw[i][2]) for i in 1:n]
    G = [Float64(X_raw[i][3]) for i in 1:n]

    model_args = (Reactant.to_rarray(M), Reactant.to_rarray(G), Reactant.to_rarray(R), n)

    selection = ProbProg.select(
        ProbProg.Address(:a),
        ProbProg.Address(:bM),
        ProbProg.Address(:bG),
        ProbProg.Address(:sigma_sq),
    )

    return (
        model_fn=model_brownian_primates_natural,
        model_args=model_args,
        selection=selection,
        position_size=4,  # a, bM, bG, sigma_sq
        model_name="Brownian motion model m14.10 (Primates301, natural MVN)",
    )
end

# init_params carry the slopes as b = [bM, bG], the data file's convention
# (the same parameter dictionaries as sicm/brownian_primates.jl).
function build_constraint(data, init_params)
    B = Float64.(data["Y"])

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
        :bM => init_b[1:1],
        :bG => init_b[2:2],
        :sigma_sq => init_sigma_sq,
        :B => B,
    )
end

function extract_samples(trace)
    bM = vec(trace.choices[:bM])
    bG = vec(trace.choices[:bG])
    return Dict{String,Any}(
        "a" => collect(vec(trace.choices[:a])),
        "b" => [[bM[i], bG[i]] for i in eachindex(bM)],
        "sigma_sq" => collect(vec(trace.choices[:sigma_sq])),
    )
end
