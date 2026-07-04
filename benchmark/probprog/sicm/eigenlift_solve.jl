# EigenLift Solve (SICM: CholeskyEigenLift, no CholeskyScale)
#
# Model:
#   alpha ~ HalfNormal(1)
#   sigma ~ HalfNormal(1)
#   f_tilde ~ Normal(0, 1)^n
#   K = alpha^2 * A_base + sigma^2 * I     ← EigenLift pattern
#   L = cholesky(K)
#   z_k = L \ x_k    for k = 1..K_SOLVE   ← tri_solve users (required by EigenLift)
#   pred = z_1 + z_2 + ... + z_K + f_tilde
#   y ~ Normal(pred, 1)
#
# A_base is a known N×N PD matrix (sample-invariant).
# x_k are known data vectors (sample-invariant).
# alpha, sigma are sampled → K = alpha^2*A + sigma^2*I is sample-dependent.
#
# SICM: **CholeskyEigenLift** on chol(alpha^2*A + sigma^2*I):
#   Hoists eigendecomposition of A (O(N^3) once).
#   Replaces per-step cholesky O(N^3) + tri_solves O(K*N^2)
#   with per-step eigenvalue shift O(N) + Q-based solves O(K*N^2).
#
# NO CholeskyScale fires (the input is s*A + t*I, not s*A).
# The benefit is entirely from EigenLift.

function model_eigenlift_solve(rng, A_base, x1, x2, x3, x4, x5, y_obs)
    n = Base.length(y_obs)

    _, alpha = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:alpha
    )

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    _, f_tilde = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde
    )

    # Covariance: alpha^2 * A_base + sigma^2 * I  → EigenLift
    alpha_2d = reshape(alpha, 1, 1)
    sigma_2d = reshape(sigma, 1, 1)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    K = (alpha_2d .* alpha_2d) .* A_base .+ (sigma_2d .* sigma_2d) .* I_n

    C = cholesky(K)

    # Full Cholesky solves: C \ x_k = K^{-1} x_k
    # Internally: UpperTriangular(U) \ (UpperTriangular(U)' \ x_k)
    # Produces two triangular_solve ops per solve — EigenLift users
    z1 = C \ x1
    z2 = C \ x2
    z3 = C \ x3
    z4 = C \ x4
    z5 = C \ x5

    pred = reshape(z1 .+ z2 .+ z3 .+ z4 .+ z5, n) .+ f_tilde

    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(pred, 1.0, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    A_raw = data["A_base"]
    A_base = [Float64(A_raw[i][j]) for i in 1:n, j in 1:n]

    xs = []
    for k in 1:5
        x_raw = data["x$k"]
        push!(xs, Reactant.to_rarray([Float64(x_raw[i][1]) for i in 1:n, j in 1:1]))
    end

    y = Float64.(data["Y"])

    A_rarray = Reactant.to_rarray(A_base)
    model_args = (A_rarray, xs[1], xs[2], xs[3], xs[4], xs[5], y)

    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:sigma),
        ProbProg.Address(:f_tilde),
    )

    return (
        model_fn = model_eigenlift_solve,
        model_args = model_args,
        selection = selection,
        position_size = 2 + n,
        model_name = "EigenLift Solve",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_sigma = Float64.(init_params["sigma"])
        init_f_tilde = Float64.(init_params["f_tilde"])
    else
        init_alpha = [1.0]
        init_sigma = [1.0]
        init_f_tilde = zeros(n)
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :sigma => init_sigma,
        :f_tilde => init_f_tilde,
        :y => y,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "alpha" => collect(vec(trace.choices[:alpha])),
        "sigma" => collect(vec(trace.choices[:sigma])),
        "f_tilde" => [collect(trace.choices[:f_tilde][i, :]) for i in 1:size(trace.choices[:f_tilde], 1)],
    )
end
