function kernel_se(X, var, length, noise; jitter=1.0e-6)
    n = Base.length(X)
    X_col = reshape(X, n, 1)
    X_row = reshape(X, 1, n)
    delta = X_col .- X_row

    length_2d = reshape(length, 1, 1)
    var_2d = reshape(var, 1, 1)
    noise_2d = reshape(noise, 1, 1)

    deltaXsq = (delta ./ length_2d) .^ 2
    K = var_2d .* exp.(-0.5 .* deltaXsq)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    K = K .+ (noise_2d .+ jitter) .* I_n
    return K
end

function model_gp(rng, X, Y)
    n = Base.length(X)
    _, kernel_var = ProbProg.sample(
        rng, ProbProg.LogNormal(0.0, 10.0, (1,)); symbol=:kernel_var
    )
    _, kernel_length = ProbProg.sample(
        rng, ProbProg.LogNormal(0.0, 10.0, (1,)); symbol=:kernel_length
    )
    _, kernel_noise = ProbProg.sample(
        rng, ProbProg.LogNormal(0.0, 10.0, (1,)); symbol=:kernel_noise
    )
    K = kernel_se(X, kernel_var, kernel_length, kernel_noise)
    _, obs = ProbProg.sample(rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:Y)
    return obs
end

function setup(data)
    n = Int(data["attrs"]["n"])
    X = Float64.(data["X"])
    Y = Float64.(data["Y"])

    X_rarray = Reactant.to_rarray(X)
    model_args = (X_rarray, Y)

    selection = ProbProg.select(
        ProbProg.Address(:kernel_var),
        ProbProg.Address(:kernel_length),
        ProbProg.Address(:kernel_noise),
    )

    return (
        model_fn = model_gp,
        model_args = model_args,
        selection = selection,
        position_size = 3,
        model_name = "Gaussian Process",
    )
end

function build_constraint(data, init_params)
    Y = Float64.(data["Y"])

    if init_params !== nothing
        init_kernel_var = Float64.(init_params["kernel_var"])
        init_kernel_length = Float64.(init_params["kernel_length"])
        init_kernel_noise = Float64.(init_params["kernel_noise"])
    else
        init_kernel_var = [1.0]
        init_kernel_length = [1.0]
        init_kernel_noise = [0.1]
    end

    return ProbProg.Constraint(
        :kernel_var => init_kernel_var,
        :kernel_length => init_kernel_length,
        :kernel_noise => init_kernel_noise,
        :Y => Y,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "kernel_var" => collect(vec(trace.choices[:kernel_var])),
        "kernel_length" => collect(vec(trace.choices[:kernel_length])),
        "kernel_noise" => collect(vec(trace.choices[:kernel_noise])),
    )
end
