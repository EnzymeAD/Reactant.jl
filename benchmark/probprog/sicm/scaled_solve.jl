# Scaled Triangular Solve (SICM: TriangularSolveScale, no Cholesky)
#
# Model:
#   sigma ~ HalfNormal(1)
#   f_tilde ~ Normal(0, 1)^n
#   z_k = (sigma * L_base) \ x_k    for k = 1..K data vectors
#   pred = z_1 + z_2 + ... + z_K + f_tilde
#   y ~ Normal(pred, 1)
#
# L_base is a known lower-triangular matrix (precomputed, sample-invariant).
# x_k are known data vectors (sample-invariant).
# sigma scales L_base (sample-dependent).
#
# SICM: **TriangularSolveScaleFactorization** on each solve:
#   tri_solve(sigma * L_base, x_k) → (1/sigma) * tri_solve(L_base, x_k)
#   tri_solve(L_base, x_k) is sample-invariant → hoisted.
#   Eliminates K × O(N^2) triangular solves per leapfrog step.
#
# NO cholesky() call in this model. The benefit is entirely from TriSolveScale.
#
# Domain: Bayesian deconvolution / inverse problems.
#   L_base is a known instrument response (e.g., point spread function decomposition).
#   sigma controls the instrument gain. Solving L\x recovers the deconvolved signal.

function model_scaled_solve(rng, L_base, x1, x2, x3, x4, x5, y_obs)
    n = Base.length(y_obs)

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    _, f_tilde = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde
    )

    # Scale the triangular factor
    sigma_2d = reshape(sigma, 1, 1)
    L_scaled = sigma_2d .* L_base    # n × n, lower triangular

    # K triangular solves: (sigma * L) \ x_k
    # TriSolveScale: → (1/sigma) * (L \ x_k), hoisting L \ x_k
    L_tri = LinearAlgebra.LowerTriangular(L_scaled)
    z1 = L_tri \ x1
    z2 = L_tri \ x2
    z3 = L_tri \ x3
    z4 = L_tri \ x4
    z5 = L_tri \ x5

    pred = reshape(z1 .+ z2 .+ z3 .+ z4 .+ z5, n) .+ f_tilde

    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(pred, 1.0, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])

    L_raw = data["L_base"]
    L_base = [Float64(L_raw[i][j]) for i in 1:n, j in 1:n]

    xs = []
    for k in 1:5
        x_raw = data["x$k"]
        push!(xs, Reactant.to_rarray([Float64(x_raw[i][1]) for i in 1:n, j in 1:1]))
    end

    y = Float64.(data["Y"])

    L_rarray = Reactant.to_rarray(L_base)
    model_args = (L_rarray, xs[1], xs[2], xs[3], xs[4], xs[5], y)

    selection = ProbProg.select(
        ProbProg.Address(:sigma),
        ProbProg.Address(:f_tilde),
    )

    return (
        model_fn = model_scaled_solve,
        model_args = model_args,
        selection = selection,
        position_size = 1 + n,
        model_name = "Scaled Solve",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_sigma = Float64.(init_params["sigma"])
        init_f_tilde = Float64.(init_params["f_tilde"])
    else
        init_sigma = [1.0]
        init_f_tilde = zeros(n)
    end

    return ProbProg.Constraint(
        :sigma => init_sigma,
        :f_tilde => init_f_tilde,
        :y => y,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "sigma" => collect(vec(trace.choices[:sigma])),
        "f_tilde" => [collect(trace.choices[:f_tilde][i, :]) for i in 1:size(trace.choices[:f_tilde], 1)],
    )
end
