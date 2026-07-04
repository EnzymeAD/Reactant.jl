# Multi-Group MVN (SICM: TriSolveScale dominates over CholeskyScale)
#
# Model:
#   sigma ~ HalfNormal(1)           (shared scale)
#   y_k ~ MVN(mu_k, sigma^2 * R)   for k = 1..K groups
#
# R is a known N x N correlation matrix (sample-invariant).
# mu_k are known group means (data, sample-invariant).
# Only sigma is sampled (position_size = 1).
#
# SICM cascade:
#   1. CholeskyScale: chol(sigma^2 * R) -> sigma * chol(R)   [saves O(N^3)]
#   2. TriSolveScale: tri_solve(sigma*L, y_k-mu_k) -> (1/sigma)*tri_solve(L, y_k-mu_k)
#      Since mu_k is invariant and y_k is data, tri_solve(L, y_k-mu_k) is invariant.
#      Hoisted K times.                                        [saves K × O(N^2)]
#   3. LogMultiply: log(diag(sigma*L)) -> log(sigma)+log(diag(L))  [saves O(N)]
#
# With K >> N, TriSolveScale dominates the total savings.
# At K=20, N=100: TriSolve saves 60% vs Cholesky 40% of total hoisted cost.
# At K=50, N=100: TriSolve saves 86% vs Cholesky 14%.
#
# Domain: Multi-site clinical trials, multi-school educational assessment.
#   K sites share a known outcome correlation structure R (from prior meta-analysis),
#   with site-specific known baselines mu_k. The shared parameter sigma controls
#   the between-subject variability common to all sites.

function model_multi_group_mvn(rng, R_data, mu_data, y_data, K, n)
    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    sigma_2d = reshape(sigma, 1, 1)
    cov = (sigma_2d .* sigma_2d) .* R_data    # sigma^2 * R

    # K independent group observations, each MVN(mu_k, sigma^2 * R)
    # Each generates a TriSolveScale opportunity
    _, obs1 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[1, :], n), cov, (n,)); symbol=:y1
    )
    _, obs2 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[2, :], n), cov, (n,)); symbol=:y2
    )
    _, obs3 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[3, :], n), cov, (n,)); symbol=:y3
    )
    _, obs4 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[4, :], n), cov, (n,)); symbol=:y4
    )
    _, obs5 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[5, :], n), cov, (n,)); symbol=:y5
    )
    _, obs6 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[6, :], n), cov, (n,)); symbol=:y6
    )
    _, obs7 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[7, :], n), cov, (n,)); symbol=:y7
    )
    _, obs8 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[8, :], n), cov, (n,)); symbol=:y8
    )
    _, obs9 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[9, :], n), cov, (n,)); symbol=:y9
    )
    _, obs10 = ProbProg.sample(
        rng, ProbProg.MultiNormal(reshape(mu_data[10, :], n), cov, (n,)); symbol=:y10
    )
    return obs1
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    K = Int(attrs["K"])

    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    mu_raw = data["mu"]
    mu = [Float64(mu_raw[k][i]) for k in 1:K, i in 1:n]
    y_raw = data["Y"]
    y_matrix = [Float64(y_raw[k][i]) for k in 1:K, i in 1:n]

    R_rarray = Reactant.to_rarray(R)
    mu_rarray = Reactant.to_rarray(mu)
    model_args = (R_rarray, mu_rarray, y_matrix, K, n)

    selection = ProbProg.select(ProbProg.Address(:sigma))

    return (
        model_fn = model_multi_group_mvn,
        model_args = model_args,
        selection = selection,
        position_size = 1,
        model_name = "Multi-Group MVN",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    K = Int(attrs["K"])

    y_raw = data["Y"]
    y_matrix = [Float64(y_raw[k][i]) for k in 1:K, i in 1:n]

    if init_params !== nothing
        init_sigma = Float64.(init_params["sigma"])
    else
        init_sigma = [1.0]
    end

    constraint_pairs = [:sigma => init_sigma]
    for k in 1:K
        push!(constraint_pairs, Symbol("y$k") => y_matrix[k, :])
    end

    return ProbProg.Constraint(constraint_pairs...)
end

function extract_samples(trace)
    return Dict{String,Any}(
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
