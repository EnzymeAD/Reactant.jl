# GP Regression with Fixed Length Scale (SICM: CholeskyEigenLift)
#
# Model: y ~ MVN(0, alpha^2 * K_base + sigma^2 * I)
#   alpha ~ HalfNormal(1)    (signal amplitude)
#   sigma ~ HalfNormal(1)    (noise std)
#   K_base = SE(X; rho_fixed) is sample-invariant
#
# SICM EigenLift pattern:
#   cholesky(s * A + t * I)  ->  eigendecomposition of A (hoisted once, O(N^3))
#   Per-iteration: d = s * lambda + t (O(N)), Q @ (Q'b / d) (O(N^2))
#   Reduces O(N^3) per iteration to O(N^2).
#
# Motivation: empirical Bayes GP, spatial models with known correlation.

function model_gp_fixed_ls(rng, x, y, rho)
    n = Base.length(x)

    _, alpha = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:alpha
    )

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    # SE kernel with fixed rho (sample-invariant)
    x_col = reshape(x, n, 1)
    x_row = reshape(x, 1, n)
    delta = x_col .- x_row
    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-0.5 .* (delta ./ rho_2d) .^ 2)

    # Covariance: alpha^2 * K_base + sigma^2 * I
    alpha_2d = reshape(alpha, 1, 1)
    sigma_2d = reshape(sigma, 1, 1)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    K = (alpha_2d .* alpha_2d) .* K_base .+ (sigma_2d .* sigma_2d) .* I_n

    _, obs = ProbProg.sample(
        rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    n = Int(data["attrs"]["n"])
    rho = Float64(data["attrs"]["rho"])

    x = Float64.(data["X"])
    y = Float64.(data["Y"])

    x_rarray = Reactant.to_rarray(x)
    rho_rarray = Reactant.to_rarray([rho])
    model_args = (x_rarray, y, rho_rarray)

    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:sigma),
    )

    return (
        model_fn = model_gp_fixed_ls,
        model_args = model_args,
        selection = selection,
        position_size = 2,
        model_name = "GP Fixed Lengthscale",
    )
end

function build_constraint(data, init_params)
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_sigma = Float64.(init_params["sigma"])
    else
        init_alpha = [1.0]
        init_sigma = [0.1]
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :sigma => init_sigma,
        :y => y,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "alpha" => collect(vec(trace.choices[:alpha])),
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
