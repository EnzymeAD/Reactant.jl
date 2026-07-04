function n_schools(
    rng, Y, sigma_data,
    state_idx, flat_district_idx, type_idx,
    n, num_states, num_districts_per_state, num_types,
    dof_baseline, scale_baseline, scale_state, scale_district, scale_type
)
    _, sigma_state = ProbProg.sample(
        rng, ProbProg.HalfCauchy(scale_state, (1,)); symbol=:sigma_state
    )
    _, sigma_district = ProbProg.sample(
        rng, ProbProg.HalfCauchy(scale_district, (1,)); symbol=:sigma_district
    )
    _, sigma_type = ProbProg.sample(
        rng, ProbProg.HalfCauchy(scale_type, (1,)); symbol=:sigma_type
    )

    _, beta_baseline = ProbProg.sample(
        rng, ProbProg.StudentT(dof_baseline, 0.0, scale_baseline, (1,)); symbol=:beta_baseline
    )

    _, beta_state = ProbProg.sample(
        rng, ProbProg.Normal(0.0, sigma_state, (num_states,)); symbol=:beta_state
    )

    num_districts_total = num_states * num_districts_per_state
    _, beta_district_flat = ProbProg.sample(
        rng, ProbProg.Normal(0.0, sigma_district, (num_districts_total,)); symbol=:beta_district_flat
    )

    _, beta_type = ProbProg.sample(
        rng, ProbProg.Normal(0.0, sigma_type, (num_types,)); symbol=:beta_type
    )

    # Vectorized index gather (like NumPyro's beta_state[state_idx])
    # Reactant compiles this to stablehlo.gather
    Yhat = beta_baseline .+ beta_state[state_idx] .+ beta_district_flat[flat_district_idx] .+ beta_type[type_idx]

    _, _ = ProbProg.sample(
        rng, ProbProg.Normal(Yhat, sigma_data, (n,)); symbol=:Y
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    num_states = Int(attrs["num_states"])
    num_districts_per_state = Int(attrs["num_districts_per_state"])
    num_types = Int(attrs["num_types"])
    dof_baseline = Float64(attrs["dof_baseline"])
    scale_baseline = Float64(attrs["scale_baseline"])
    scale_state = Float64(attrs["scale_state"])
    scale_district = Float64(attrs["scale_district"])
    scale_type = Float64(attrs["scale_type"])

    Y = Float64.(data["Y"])
    sigma_data = Float64.(data["sigma"])

    # Indices: 0-based in data, convert to 1-based Julia
    state_idx = Int.(attrs["state_idx"]) .+ 1
    district_idx = Int.(attrs["district_idx"]) .+ 1
    type_idx = Int.(attrs["type_idx"]) .+ 1

    # Precompute flat district index: (state-1)*num_districts_per_state + district
    flat_district_idx = (state_idx .- 1) .* num_districts_per_state .+ district_idx

    Y_rarray = Reactant.to_rarray(Y)
    sigma_rarray = Reactant.to_rarray(sigma_data)
    state_idx_rarray = Reactant.to_rarray(state_idx)
    flat_district_idx_rarray = Reactant.to_rarray(flat_district_idx)
    type_idx_rarray = Reactant.to_rarray(type_idx)

    model_args = (
        Y_rarray, sigma_rarray,
        state_idx_rarray, flat_district_idx_rarray, type_idx_rarray,
        n, num_states, num_districts_per_state, num_types,
        dof_baseline, scale_baseline, scale_state, scale_district, scale_type,
    )

    selection = ProbProg.select(
        ProbProg.Address(:sigma_state),
        ProbProg.Address(:sigma_district),
        ProbProg.Address(:sigma_type),
        ProbProg.Address(:beta_baseline),
        ProbProg.Address(:beta_state),
        ProbProg.Address(:beta_district_flat),
        ProbProg.Address(:beta_type),
    )

    # 3 sigmas + 1 baseline + num_states + num_states*num_districts + num_types
    num_districts_total = num_states * num_districts_per_state
    position_size = 3 + 1 + num_states + num_districts_total + num_types

    return (
        model_fn = n_schools,
        model_args = model_args,
        selection = selection,
        position_size = position_size,
        model_name = "N Schools",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    num_states = Int(attrs["num_states"])
    num_districts_per_state = Int(attrs["num_districts_per_state"])
    num_types = Int(attrs["num_types"])
    Y = Float64.(data["Y"])

    num_districts_total = num_states * num_districts_per_state

    if init_params !== nothing
        init_sigma_state = Float64.(init_params["sigma_state"])
        init_sigma_district = Float64.(init_params["sigma_district"])
        init_sigma_type = Float64.(init_params["sigma_type"])
        init_beta_baseline = Float64.(init_params["beta_baseline"])
        init_beta_state = Float64.(init_params["beta_state"])
        init_beta_district_flat = Float64.(init_params["beta_district_flat"])
        init_beta_type = Float64.(init_params["beta_type"])
    else
        init_sigma_state = [1.0]
        init_sigma_district = [1.0]
        init_sigma_type = [1.0]
        init_beta_baseline = [0.0]
        init_beta_state = zeros(num_states)
        init_beta_district_flat = zeros(num_districts_total)
        init_beta_type = zeros(num_types)
    end

    return ProbProg.Constraint(
        :sigma_state => init_sigma_state,
        :sigma_district => init_sigma_district,
        :sigma_type => init_sigma_type,
        :beta_baseline => init_beta_baseline,
        :beta_state => init_beta_state,
        :beta_district_flat => init_beta_district_flat,
        :beta_type => init_beta_type,
        :Y => Y,
    )
end

function extract_samples(trace)
    sigma_state_samples = vec(trace.choices[:sigma_state])
    sigma_district_samples = vec(trace.choices[:sigma_district])
    sigma_type_samples = vec(trace.choices[:sigma_type])
    beta_baseline_samples = vec(trace.choices[:beta_baseline])
    beta_state_samples = trace.choices[:beta_state]
    beta_district_flat_samples = trace.choices[:beta_district_flat]
    beta_type_samples = trace.choices[:beta_type]

    return Dict{String,Any}(
        "sigma_state" => collect(sigma_state_samples),
        "sigma_district" => collect(sigma_district_samples),
        "sigma_type" => collect(sigma_type_samples),
        "beta_baseline" => collect(beta_baseline_samples),
        "beta_state" => [collect(beta_state_samples[i, :]) for i in 1:size(beta_state_samples, 1)],
        "beta_district_flat" => [collect(beta_district_flat_samples[i, :]) for i in 1:size(beta_district_flat_samples, 1)],
        "beta_type" => [collect(beta_type_samples[i, :]) for i in 1:size(beta_type_samples, 1)],
    )
end
