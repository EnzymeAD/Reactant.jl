using PythonCall, CondaPkg

if !@isdefined(_NUMPYRO_AVAILABLE)
    const _NUMPYRO_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
end

function check_numpyro_available()
    if _NUMPYRO_AVAILABLE[] !== nothing
        return _NUMPYRO_AVAILABLE[]
    end
    try
        CondaPkg.add_pip("jax"; version="==0.9.0")
        CondaPkg.add_pip("numpyro"; version="==0.19.0")

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
        kernel; num_warmup=num_warmup, num_samples=num_samples, progress_bar=false
    )

    mcmc.run(rng_key, model_args...; model_kwargs..., init_params=init_params)

    samples_py = mcmc.get_samples()
    result = Dict{Symbol,Vector{Float64}}()
    for name in param_names
        arr = pyconvert(Vector{Float64}, np.asarray(samples_py[string(name)]).flatten())
        result[name] = arr
    end
    return result
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
