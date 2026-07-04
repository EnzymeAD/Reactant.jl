# Multi-Task Regression with K=10 tasks sharing parameter-dependent covariance.
# Same structure as multi_task_regression.jl but with 10 tasks to amplify
# the cross-site adjoint overhead.

function multi_task_regression_k10(rng, X, Sigma_base, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10)
    d = size(X, 2)
    n = size(X, 1)

    _, tau = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau)
    tau_2d = reshape(tau, 1, 1)
    I_d = Matrix{Float64}(LinearAlgebra.I, d, d)
    Sigma = (tau_2d .* tau_2d) .* Sigma_base .+ I_d

    _, beta1 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta1)
    _, sigma1 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma1)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta1, d, 1), n), sigma1, (n,)); symbol=:y1)

    _, beta2 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta2)
    _, sigma2 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma2)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta2, d, 1), n), sigma2, (n,)); symbol=:y2)

    _, beta3 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta3)
    _, sigma3 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma3)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta3, d, 1), n), sigma3, (n,)); symbol=:y3)

    _, beta4 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta4)
    _, sigma4 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma4)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta4, d, 1), n), sigma4, (n,)); symbol=:y4)

    _, beta5 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta5)
    _, sigma5 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma5)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta5, d, 1), n), sigma5, (n,)); symbol=:y5)

    _, beta6 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta6)
    _, sigma6 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma6)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta6, d, 1), n), sigma6, (n,)); symbol=:y6)

    _, beta7 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta7)
    _, sigma7 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma7)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta7, d, 1), n), sigma7, (n,)); symbol=:y7)

    _, beta8 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta8)
    _, sigma8 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma8)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta8, d, 1), n), sigma8, (n,)); symbol=:y8)

    _, beta9 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta9)
    _, sigma9 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma9)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta9, d, 1), n), sigma9, (n,)); symbol=:y9)

    _, beta10 = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(d), Sigma, (d,)); symbol=:beta10)
    _, sigma10 = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma10)
    _, _ = ProbProg.sample(rng, ProbProg.Normal(reshape(X * reshape(beta10, d, 1), n), sigma10, (n,)); symbol=:y10)

    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    d = Int(attrs["d"])
    K = 10

    X_raw = data["X"]
    X = [Float64(X_raw[i][j]) for i in 1:n, j in 1:d]
    Sigma_raw = data["Sigma_prior"]
    Sigma_base = [Float64(Sigma_raw[i][j]) for i in 1:d, j in 1:d]

    ys = Tuple(Float64.(data["Y$k"]) for k in 1:K)

    X_rarray = Reactant.to_rarray(X)
    Sigma_rarray = Reactant.to_rarray(Sigma_base)
    model_args = (X_rarray, Sigma_rarray, ys...)

    sel_addrs = [ProbProg.Address(:tau)]
    for k in 1:K
        push!(sel_addrs, ProbProg.Address(Symbol("beta$k")))
        push!(sel_addrs, ProbProg.Address(Symbol("sigma$k")))
    end
    selection = ProbProg.select(sel_addrs...)

    return (
        model_fn = multi_task_regression_k10,
        model_args = model_args,
        selection = selection,
        position_size = 1 + K * (d + 1),
        model_name = "Multi-Task Regression K=10",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    d = Int(attrs["d"])
    K = 10

    pairs = Pair{Symbol,Vector{Float64}}[:tau => init_params !== nothing ? Float64.(init_params["tau"]) : [1.0]]
    for k in 1:K
        bk = init_params !== nothing ? Float64.(init_params["beta$k"]) : zeros(d)
        sk = init_params !== nothing ? Float64.(init_params["sigma$k"]) : [1.0]
        push!(pairs, Symbol("beta$k") => bk)
        push!(pairs, Symbol("sigma$k") => sk)
    end
    for k in 1:K
        push!(pairs, Symbol("y$k") => Float64.(data["Y$k"]))
    end
    return ProbProg.Constraint(pairs...)
end

function extract_samples(trace)
    result = Dict{String,Any}("tau" => collect(vec(trace.choices[:tau])))
    for k in 1:10
        sym = Symbol("beta$k")
        result["beta$k"] = [collect(trace.choices[sym][i, :]) for i in 1:size(trace.choices[sym], 1)]
        result["sigma$k"] = collect(vec(trace.choices[Symbol("sigma$k")]))
    end
    return result
end
