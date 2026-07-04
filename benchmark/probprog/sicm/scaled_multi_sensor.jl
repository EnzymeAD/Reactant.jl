# Scaled Multi-Sensor Fusion (SICM: DotGeneralScaleFactorization)
#
# Model:
#   sigma ~ HalfNormal(1)            (global gain)
#   f_tilde ~ Normal(0, 1)^D_obs     (latent noise, NCP)
#   z_k = (sigma * M_k) @ s_k        for k = 1..K sensors
#   pred = z_1 + z_2 + ... + z_K
#   y ~ Normal(pred + f_tilde, 1)
#
# M_k (D_obs × D_state) are known sensor measurement matrices (invariant)
# s_k (D_state × 1) are known calibration signals (invariant)
# sigma is the unknown shared gain parameter (sample-dependent)
#
# SICM: **DotGeneralScaleFactorization** on each (sigma * M_k) @ s_k
#   (sigma * M_k) @ s_k  →  sigma * (M_k @ s_k)
#   M_k @ s_k is sample-invariant → hoisted once outside MCMC loop.
#   Eliminates K × O(D²) matvecs per leapfrog step.
#
# Domain: Sensor fusion / calibration.
#   In array signal processing, multiple sensors share a common electronic
#   gain that drifts over time. The measurement matrices M_k encode each
#   sensor's spatial response (beam pattern, antenna gain, etc.), which is
#   fixed for a given hardware configuration. Inferring the shared gain
#   from multiple simultaneous observations is a standard calibration task.

function model_scaled_multi_sensor(rng, M1, M2, M3, M4, M5, s1, s2, s3, s4, s5, y_obs)
    n = Base.length(y_obs)

    _, sigma = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma
    )

    _, f_tilde = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (n,)); symbol=:f_tilde
    )

    # Each sensor projection: (sigma * M_k) @ s_k
    # DotGeneralScale: → sigma * (M_k @ s_k), hoists M_k @ s_k
    sigma_2d = reshape(sigma, 1, 1)
    z1 = (sigma_2d .* M1) * s1
    z2 = (sigma_2d .* M2) * s2
    z3 = (sigma_2d .* M3) * s3
    z4 = (sigma_2d .* M4) * s4
    z5 = (sigma_2d .* M5) * s5

    pred = reshape(z1 .+ z2 .+ z3 .+ z4 .+ z5, n)
    mu = pred .+ f_tilde

    _, obs = ProbProg.sample(
        rng, ProbProg.Normal(mu, 1.0, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    d_state = Int(attrs["d_state"])

    # Load sensor matrices and calibration signals
    Ms = []
    ss_vecs = []
    for k in 1:5
        M_raw = data["M$k"]
        push!(Ms, Reactant.to_rarray([Float64(M_raw[i][j]) for i in 1:n, j in 1:d_state]))
        s_raw = data["s$k"]
        push!(ss_vecs, Reactant.to_rarray([Float64(s_raw[i][1]) for i in 1:d_state, j in 1:1]))
    end

    y = Float64.(data["Y"])

    model_args = (Ms[1], Ms[2], Ms[3], Ms[4], Ms[5],
                  ss_vecs[1], ss_vecs[2], ss_vecs[3], ss_vecs[4], ss_vecs[5],
                  y)

    selection = ProbProg.select(
        ProbProg.Address(:sigma),
        ProbProg.Address(:f_tilde),
    )

    return (
        model_fn = model_scaled_multi_sensor,
        model_args = model_args,
        selection = selection,
        position_size = 1 + n,
        model_name = "Scaled Multi-Sensor",
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
