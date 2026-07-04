# Turing.jl model spec for N Schools
# Matches NumPyro n_schools.py and Impulse standard/n_schools.jl

using Turing, Distributions, LinearAlgebra

@model function turing_n_schools(
    Y, sigma_data, state_idx, district_idx, type_idx,
    n, num_states, num_districts_per_state, num_types,
    dof_baseline, scale_baseline, scale_state, scale_district, scale_type
)
    sigma_state ~ truncated(Cauchy(0.0, scale_state), 0.0, Inf)
    sigma_district ~ truncated(Cauchy(0.0, scale_district), 0.0, Inf)
    sigma_type ~ truncated(Cauchy(0.0, scale_type), 0.0, Inf)

    beta_baseline ~ LocationScale(0.0, scale_baseline, TDist(dof_baseline))

    beta_state ~ filldist(Normal(0.0, sigma_state), num_states)

    num_districts_total = num_states * num_districts_per_state
    beta_district_flat ~ filldist(Normal(0.0, sigma_district), num_districts_total)

    beta_type ~ filldist(Normal(0.0, sigma_type), num_types)

    flat_di = (state_idx .- 1) .* num_districts_per_state .+ district_idx
    Yhat = beta_baseline .+ beta_state[state_idx] .+ beta_district_flat[flat_di] .+ beta_type[type_idx]
    Turing.@addlogprob! sum(logpdf.(Normal.(Yhat, sigma_data), Y))
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

    # Indices: 0-based in JSON → 1-based for Julia
    state_idx = Int.(attrs["state_idx"]) .+ 1
    district_idx = Int.(attrs["district_idx"]) .+ 1
    type_idx = Int.(attrs["type_idx"]) .+ 1

    model = turing_n_schools(
        Y, sigma_data, state_idx, district_idx, type_idx,
        n, num_states, num_districts_per_state, num_types,
        dof_baseline, scale_baseline, scale_state, scale_district, scale_type,
    )

    return (
        turing_model = model,
        model_name = "N Schools",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    sigma_state = Array(chain[:sigma_state])[start:total]
    sigma_district = Array(chain[:sigma_district])[start:total]
    sigma_type = Array(chain[:sigma_type])[start:total]
    beta_baseline = Array(chain[:beta_baseline])[start:total]

    num_states = length([s for s in names(chain, :parameters) if startswith(string(s), "beta_state[")])
    beta_state = hcat([Array(chain[Symbol("beta_state[$i]")])[start:total] for i in 1:num_states]...)

    num_districts_total = length([s for s in names(chain, :parameters) if startswith(string(s), "beta_district_flat[")])
    beta_district_flat = hcat([Array(chain[Symbol("beta_district_flat[$i]")])[start:total] for i in 1:num_districts_total]...)

    num_types = length([s for s in names(chain, :parameters) if startswith(string(s), "beta_type[")])
    beta_type = hcat([Array(chain[Symbol("beta_type[$i]")])[start:total] for i in 1:num_types]...)

    return Dict{String,Any}(
        "sigma_state" => collect(sigma_state),
        "sigma_district" => collect(sigma_district),
        "sigma_type" => collect(sigma_type),
        "beta_baseline" => collect(beta_baseline),
        "beta_state" => [collect(beta_state[i, :]) for i in axes(beta_state, 1)],
        "beta_district_flat" => [collect(beta_district_flat[i, :]) for i in axes(beta_district_flat, 1)],
        "beta_type" => [collect(beta_type[i, :]) for i in axes(beta_type, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end

    sigma_state = Float64(init_params["sigma_state"][1])
    sigma_district = Float64(init_params["sigma_district"][1])
    sigma_type = Float64(init_params["sigma_type"][1])
    beta_baseline = Float64(init_params["beta_baseline"][1])
    beta_state = Float64.(init_params["beta_state"])
    beta_district_flat = Float64.(init_params["beta_district_flat"])
    beta_type = Float64.(init_params["beta_type"])

    return (
        sigma_state=sigma_state,
        sigma_district=sigma_district,
        sigma_type=sigma_type,
        beta_baseline=beta_baseline,
        beta_state=beta_state,
        beta_district_flat=beta_district_flat,
        beta_type=beta_type,
    )
end
