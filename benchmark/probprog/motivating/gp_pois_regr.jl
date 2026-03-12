# GP Poisson Regression (posteriordb gp_pois_regr)
# Non-centered parameterization, fixed rho, no jitter (enables SICM rewrites).

function model_gp_pois_regr(rng, x, k, rho)
    n = Base.length(x)

    _, alpha = ProbProg.sample(
        rng, ProbProg.HalfNormal(2.0, (1,)); symbol=:alpha
    )

    _, f_tilde = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde
    )

    # SE kernel with fixed rho
    x_col = reshape(x, n, 1)
    x_row = reshape(x, 1, n)
    delta = x_col .- x_row
    rho_2d = reshape(rho, 1, 1)
    K_base = exp.(-0.5 .* (delta ./ rho_2d) .^ 2)

    # NCP: cov = alpha^2 * K_base, no jitter
    alpha_2d = reshape(alpha, 1, 1)
    cov = (alpha_2d .* alpha_2d) .* K_base

    C = cholesky(cov)
    f = reshape(C.factors' * reshape(f_tilde, n, 1), n)

    _, obs = ProbProg.sample(
        rng, ProbProg.Poisson(exp.(f), (n,)); symbol=:k
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    rho = Float64(attrs["rho"])

    x = Float64.(data["X"])
    k = Float64.(data["Y"])

    x_rarray = Reactant.to_rarray(x)
    k_rarray = Reactant.to_rarray(k)
    rho_rarray = Reactant.to_rarray([rho])
    model_args = (x_rarray, k_rarray, rho_rarray)

    selection = ProbProg.select(ProbProg.Address(:alpha), ProbProg.Address(:f_tilde))

    return (
        model_fn = model_gp_pois_regr,
        model_args = model_args,
        selection = selection,
        position_size = 1 + n,
        model_name = "GP Poisson Regression",
    )
end

function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    k = Float64.(data["Y"])

    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_f_tilde = Float64.(init_params["f_tilde"])
    else
        init_alpha = [1.0]
        init_f_tilde = zeros(n)
    end

    return ProbProg.Constraint(
        :alpha => init_alpha,
        :f_tilde => init_f_tilde,
        :k => k,
    )
end

function extract_samples(trace)
    alpha_samples = vec(trace.choices[:alpha])
    f_tilde_samples = trace.choices[:f_tilde]
    return Dict{String,Any}(
        "alpha" => collect(alpha_samples),
        "f_tilde" => [collect(f_tilde_samples[i, :]) for i in 1:size(f_tilde_samples, 1)],
    )
end
