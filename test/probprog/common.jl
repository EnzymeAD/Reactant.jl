using PythonCall

if !@isdefined(_JAX_AVAILABLE)
    const _JAX_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
end

function check_jax_available()
    if _JAX_AVAILABLE[] !== nothing
        return _JAX_AVAILABLE[]
    end
    if lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())) != "cpu"
        @warn "Skipping JAX comparison tests on non-CPU backend"
        _JAX_AVAILABLE[] = false
        return false
    end
    try
        os = pyimport("os")
        os.environ.__setitem__("JAX_ENABLE_X64", "1")
        jax = pyimport("jax")
        jax.config.update("jax_enable_x64", true)
        _JAX_AVAILABLE[] = true
    catch e
        @warn "JAX not available, skipping pointwise comparison tests" exception = e
        _JAX_AVAILABLE[] = false
    end
    return _JAX_AVAILABLE[]
end

if !@isdefined(_NUMPYRO_AVAILABLE)
    const _NUMPYRO_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
end

function check_numpyro_available()
    if _NUMPYRO_AVAILABLE[] !== nothing
        return _NUMPYRO_AVAILABLE[]
    end
    if lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend())) != "cpu"
        @warn "Skipping NumPyro comparison tests on non-CPU backend"
        _NUMPYRO_AVAILABLE[] = false
        return false
    end
    try
        os = pyimport("os")
        os.environ.__setitem__("JAX_ENABLE_X64", "1")
        jax = pyimport("jax")
        jax.config.update("jax_enable_x64", true)

        pyimport("numpyro")
        _NUMPYRO_AVAILABLE[] = true
    catch e
        @warn "NumPyro not available, skipping comparison tests" exception = e
        _NUMPYRO_AVAILABLE[] = false
    end
    return _NUMPYRO_AVAILABLE[]
end

function seed_to_rbg_key(seed::Vector{UInt64})
    np = pyimport("numpy")
    jax_random = pyimport("jax.random")
    seed_np = np.array(pylist([Int(s) for s in seed]); dtype=np.uint64)
    seed_u32 = seed_np.view(np.uint32)
    return jax_random.wrap_key_data(seed_u32; impl="rbg")
end

function run_numpyro_mcmc(;
    model,
    rng_key,
    model_args::Tuple,
    model_kwargs::NamedTuple=(;),
    init_params,
    param_names::Vector{Symbol},
    algorithm::Symbol=:NUTS,
    num_warmup::Int=200,
    num_samples::Int=5,
    step_size::Float64=0.1,
    max_tree_depth::Int=10,
    trajectory_length::Union{Nothing,Float64}=nothing,
    dense_mass::Bool=true,
    inverse_mass_matrix=nothing,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    num_chains::Int=1,
    return_last_key::Bool=false,
)
    numpyro_infer = pyimport("numpyro.infer")
    np = pyimport("numpy")

    if algorithm == :NUTS
        kernel = numpyro_infer.NUTS(
            model;
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            inverse_mass_matrix=inverse_mass_matrix,
            find_heuristic_step_size=false,
        )
    elseif algorithm == :HMC
        tl = something(trajectory_length, step_size * 10)
        kernel = numpyro_infer.HMC(
            model;
            step_size=step_size,
            trajectory_length=tl,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            inverse_mass_matrix=inverse_mass_matrix,
            find_heuristic_step_size=false,
        )
    else
        error("Unknown algorithm: $algorithm")
    end

    mcmc = numpyro_infer.MCMC(
        kernel;
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=false,
    )

    mcmc.run(rng_key, model_args...; model_kwargs..., init_params=init_params)

    # rbg keys are uint32[..., 2, 2]; view them as two uint64 per chain to compare bitwise.
    last_keys = if return_last_key
        jax_random = pyimport("jax.random")
        key = mcmc.last_state.rng_key
        kd = np.asarray(jax_random.key_data(key)).view(np.uint64)
        if num_chains > 1
            pyconvert(Matrix{UInt64}, kd.reshape(num_chains, 2))
        else
            pyconvert(Vector{UInt64}, kd.reshape(2))
        end
    else
        nothing
    end

    result = if num_chains > 1
        samples_py = mcmc.get_samples(; group_by_chain=true)
        d = Dict{Symbol,Matrix{Float64}}()
        for name in param_names
            arr2d = np.asarray(samples_py[string(name)]).reshape(num_chains, num_samples)
            d[name] = pyconvert(Matrix{Float64}, arr2d)
        end
        d
    else
        samples_py = mcmc.get_samples()
        d = Dict{Symbol,Vector{Float64}}()
        for name in param_names
            d[name] = pyconvert(Vector{Float64}, np.asarray(samples_py[string(name)]).flatten())
        end
        d
    end

    return return_last_key ? (result, last_keys) : result
end

function run_numpyro_svi(;
    model,
    rng_key,
    model_args::Tuple=(),
    model_kwargs::NamedTuple=(;),
    init_loc_values::Union{Nothing,AbstractDict}=nothing,
    init_scale::Float64=0.1,
    num_steps::Int=5,
    step_size::Float64=0.01,
    optimizer::Symbol=:Adam,
    num_particles::Int=1,
    adam_b1::Float64=0.9,
    adam_b2::Float64=0.999,
    adam_eps::Float64=1.0e-8,
    clip_norm::Float64=10.0,
    momentum_mass::Float64=0.9,
    rmsprop_gamma::Float64=0.9,
    rmsprop_eps::Float64=1.0e-8,
    guide_kind::Symbol=:AutoNormal,
    stable_update::Bool=false,
    param_names::Vector{Symbol},
)
    numpyro_infer = pyimport("numpyro.infer")
    numpyro_optim = pyimport("numpyro.optim")
    numpyro_init = pyimport("numpyro.infer.initialization")
    autoguide_mod = pyimport("numpyro.infer.autoguide")
    np = pyimport("numpy")

    init_loc_fn = if init_loc_values === nothing
        nothing
    else
        py_dict = pydict()
        for (k, v) in init_loc_values
            py_dict[string(k)] = v
        end
        numpyro_init.init_to_value(; values=py_dict)
    end

    guide = if guide_kind == :AutoNormal
        if init_loc_fn === nothing
            autoguide_mod.AutoNormal(model; init_scale=init_scale)
        else
            autoguide_mod.AutoNormal(model; init_loc_fn=init_loc_fn, init_scale=init_scale)
        end
    elseif guide_kind == :AutoDelta
        if init_loc_fn === nothing
            autoguide_mod.AutoDelta(model)
        else
            autoguide_mod.AutoDelta(model; init_loc_fn=init_loc_fn)
        end
    else
        error("Unknown guide_kind: $guide_kind. Supported: :AutoNormal, :AutoDelta")
    end

    if optimizer == :Adam
        opt = numpyro_optim.Adam(;
            step_size=step_size, b1=adam_b1, b2=adam_b2, eps=adam_eps
        )
    elseif optimizer == :SGD
        opt = numpyro_optim.SGD(; step_size=step_size)
    elseif optimizer == :ClippedAdam
        opt = numpyro_optim.ClippedAdam(;
            step_size=step_size, b1=adam_b1, b2=adam_b2, eps=adam_eps, clip_norm=clip_norm
        )
    elseif optimizer == :Momentum
        opt = numpyro_optim.Momentum(; step_size=step_size, mass=momentum_mass)
    elseif optimizer == :RMSProp
        opt = numpyro_optim.RMSProp(;
            step_size=step_size, gamma=rmsprop_gamma, eps=rmsprop_eps
        )
    elseif optimizer == :Adagrad
        opt = numpyro_optim.Adagrad(; step_size=step_size, momentum=momentum_mass)
    elseif optimizer == :RMSPropMomentum
        opt = numpyro_optim.RMSPropMomentum(;
            step_size=step_size,
            gamma=rmsprop_gamma,
            eps=rmsprop_eps,
            momentum=momentum_mass,
        )
    else
        error("Unknown optimizer: $optimizer")
    end

    elbo = numpyro_infer.Trace_ELBO(; num_particles=num_particles)
    svi = numpyro_infer.SVI(model, guide, opt; loss=elbo)

    state = svi.init(rng_key, model_args...; model_kwargs...)

    trajectory = Dict(name => Float64[] for name in param_names)
    elbo_traj = Float64[]

    for _ in 1:num_steps
        out = if stable_update
            svi.stable_update(state, model_args...; model_kwargs...)
        else
            svi.update(state, model_args...; model_kwargs...)
        end
        state = out[0]
        loss = out[1]
        push!(elbo_traj, -pyconvert(Float64, np.asarray(loss).item()))
        params = svi.get_params(state)
        for name in param_names
            arr = np.asarray(params[string(name)]).flatten()
            v = pyconvert(Float64, arr[0].item())
            push!(trajectory[name], v)
        end
    end

    return (loc_traj=trajectory, elbo_traj=elbo_traj)
end

function compare_trajectories_pointwise(
    impulse_traj::AbstractMatrix,
    numpyro_traj::Vector{Float64};
    atol::Float64=1e-10,
    rtol::Float64=1e-10,
    label::String="trajectory",
)
    @testset "NumPyro pointwise comparison: $label" begin
        impulse_flat = vec(impulse_traj)
        @assert length(impulse_flat) == length(numpyro_traj) (
            "size mismatch: impulse=$(length(impulse_flat)) numpyro=$(length(numpyro_traj))"
        )
        max_abs = maximum(abs.(impulse_flat .- numpyro_traj))
        @info "$label max abs diff = $max_abs (atol=$atol)"
        @test impulse_flat ≈ numpyro_traj atol = atol rtol = rtol
    end
end

function compare_samples_pointwise(
    reactant_trace,
    numpyro_samples::Dict{Symbol,Vector{Float64}},
    param_names::Vector{Symbol};
    atol::Float64=1e-8,
    rtol::Float64=1e-6,
)
    @testset "NumPyro pointwise comparison" begin
        for name in param_names
            reactant_vals = vec(reactant_trace.choices[name])
            numpyro_vals = numpyro_samples[name]
            max_abs_diff = maximum(abs.(reactant_vals .- numpyro_vals))
            max_rel_diff = maximum(
                abs.(reactant_vals .- numpyro_vals) ./ max.(abs.(numpyro_vals), 1e-300)
            )
            @test reactant_vals ≈ numpyro_vals atol = atol rtol = rtol
        end
    end
end

# Test-only reference samplers/logpdfs: the oracles for the emitted distributions.

using Random: randexp
using LinearAlgebra: cholesky, I

const _refOps = Reactant.Ops

ref_normal_sampler(rng, μ, σ, shape) = μ .+ σ .* randn(rng, shape)
function ref_normal_logpdf(x, μ, σ, _)
    z = (x .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ))
end

ref_exponential_sampler(rng, λ, shape) = randexp(rng, shape) ./ λ
ref_exponential_logpdf(x, λ, _) = sum(log.(λ) .- λ .* x)

ref_halfnormal_sampler(rng, σ, shape) = abs.(σ .* randn(rng, shape))
function ref_halfnormal_logpdf(x, σ, _)
    n = length(x)
    log_sigma = sum(log.(σ))
    return n * log(2) - n * log_sigma - n / 2 * log(2π) - sum(x .* x ./ (2 .* σ .* σ))
end

ref_lognormal_sampler(rng, μ, σ, shape) = exp.(μ .+ σ .* randn(rng, shape))
function ref_lognormal_logpdf(x, μ, σ, _)
    log_x = log.(x)
    z = (log_x .- μ) ./ σ
    return sum(.-(abs2.(z) .+ log(2π)) ./ 2 .- log.(σ) .- log_x)
end

function ref_bernoulli_sampler(rng, logits, shape)
    probs = 1.0 ./ (1.0 .+ exp.(-logits))
    u = rand(rng, size(probs)...)
    return Float64.(u .< probs)
end
function ref_bernoulli_logpdf(y, logits, _)
    return sum(y .* logits .- max.(logits, 0.0) .- log1p.(exp.(.-abs.(logits))))
end

function ref_discreteuniform_sampler(rng, low, high, shape)
    n = (high - low) + 1.0
    u = rand(rng, shape...)
    return low .+ floor.(u .* n)
end
function ref_discreteuniform_logpdf(x, low, high, _)
    logp = .-log.((high + 1.0) - low)
    return sum(logp .+ zero.(x))
end

function ref_halfcauchy_sampler(rng, scale, shape)
    u = rand(rng, shape...)
    finfo_eps = eps(Float64)
    u_cauchy = u .* (1.0 - finfo_eps) .+ finfo_eps
    return scale .* abs.(tan.(π .* (u_cauchy .- 0.5)))
end
function ref_halfcauchy_logpdf(x, scale, _)
    return sum(log(2 / π) .- log.(scale) .- log1p.((x ./ scale) .* (x ./ scale)))
end

function ref_mvn_sampler(rng, μ, Σ, shape)
    n = shape[1]
    z = randn(rng, shape)
    C = cholesky(Σ)
    return μ .+ reshape(C.factors' * reshape(z, n, 1), n)
end
function ref_mvn_logpdf(x, μ, Σ, shape)
    n = shape[1]
    C = cholesky(Σ)
    diff = x .- μ
    alpha = reshape(C \ reshape(diff, n, 1), n)
    I_n = Matrix{Float64}(I, n, n)
    U_diag = vec(sum(C.factors .* I_n; dims=1))
    log_det = 2.0 * sum(log.(U_diag))
    return -0.5 * (sum(diff .* alpha) + log_det + n * log(2π))
end

function ref_gamma_logpdf(x, concentration, rate, _)
    α = _refOps.fill(0.0, size(x)...) .+ concentration
    β = _refOps.fill(0.0, size(x)...) .+ rate
    return sum((α .- 1) .* log.(x) .- β .* x .- _refOps.lgamma(α) .+ α .* log.(β))
end

function ref_studentt_logpdf(x, df, loc, scale, _)
    df_t = _refOps.fill(0.0, size(x)...) .+ df
    scale_t = _refOps.fill(0.0, size(x)...) .+ scale
    y = (x .- loc) ./ scale_t
    return sum(
        _refOps.lgamma((df_t .+ 1) ./ 2) .- _refOps.lgamma(df_t ./ 2) .-
        0.5 .* log.(df_t .* π) .- log.(scale_t) .-
        (df_t .+ 1) ./ 2 .* log1p.(y .* y ./ df_t),
    )
end

function ref_poisson_logpdf(k, rate, _)
    return sum(k .* log.(rate) .- rate .- _refOps.lgamma(k .+ 1.0))
end

function ref_dirichlet_logpdf(x, alpha, _)
    α = _refOps.fill(0.0, size(x)...) .+ alpha
    d = ndims(α)
    normalize_term = sum(_refOps.lgamma(α); dims=d) .- _refOps.lgamma(sum(α; dims=d))
    return sum(sum(log.(x) .* (α .- 1.0); dims=d) .- normalize_term)
end
