import ..Reactant
using ..Reactant: TracedRArray, ReactantRNG, ConcreteRNumber
using Serialization: Serialization

# Mirrors mlir::impulse::buildStateSchema; the op verifier rejects any other order.
const _STATE_SLOTS_PLAIN = String[
    "position", "gradient", "potential_energy", "step_size", "inverse_mass_matrix"
]
const _STATE_SLOTS_ADAPT = String[
    "da_log_step_size",
    "da_log_step_size_avg",
    "da_gradient_avg",
    "da_step_count",
    "da_prox_center",
    "welford_mean",
    "welford_m2",
    "welford_n",
    "window_idx",
    "warmup_offset",
]

function state_schema_names(adaptive::Bool, num_chains::Int=1)
    names =
        adaptive ? vcat(_STATE_SLOTS_PLAIN, _STATE_SLOTS_ADAPT) : copy(_STATE_SLOTS_PLAIN)
    num_chains > 1 && push!(names, "rng")
    return names
end

function _state_schema_attr(names)
    return MLIR.IR.Attribute(MLIR.IR.Attribute[MLIR.IR.Attribute(n) for n in names])
end

# Must agree with the diagonal test in buildStateSchema.
function _mass_is_diagonal(inv_mass_shape)
    n = length(inv_mass_shape)
    return n == 1 || (n == 2 && inv_mass_shape[1] == 1)
end

function state_out_types(
    adaptive::Bool,
    position_type,
    scalar_type,
    inv_mass_type,
    pos_size::Int,
    diagonal::Bool;
    num_chains::Int=1,
    inv_mass_shape=nothing,
    rng_shape=nothing,
)
    if num_chains > 1
        f64 = MLIR.IR.Type(Float64)
        i64 = MLIR.IR.Type(Int64)
        ui64 = MLIR.IR.Type(UInt64)
        _cm(shape, et) = MLIR.IR.TensorType(Int[num_chains, shape...], et)
        pos_t = _cm((1, pos_size), f64)
        scal_t = _cm((), f64)
        scal_i64 = _cm((), i64)
        imm_t = _cm(Tuple(inv_mass_shape), f64)
        types = MLIR.IR.Type[pos_t, pos_t, scal_t, scal_t, imm_t]
        if adaptive
            welford_mean_t = _cm((pos_size,), f64)
            welford_m2_t = diagonal ? _cm((pos_size,), f64) : _cm((pos_size, pos_size), f64)
            append!(
                types,
                MLIR.IR.Type[
                    scal_t,
                    scal_t,
                    scal_t,
                    scal_i64,
                    scal_t,
                    welford_mean_t,
                    welford_m2_t,
                    scal_i64,
                    scal_i64,
                    scal_i64,
                ],
            )
        end
        push!(types, _cm(Tuple(rng_shape), ui64))
        return types
    end
    types = MLIR.IR.Type[
        position_type, position_type, scalar_type, scalar_type, inv_mass_type
    ]
    if adaptive
        f64 = MLIR.IR.Type(Float64)
        scalar_i64_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Int64))
        welford_mean_type = MLIR.IR.TensorType([pos_size], f64)
        welford_m2_type = if diagonal
            MLIR.IR.TensorType([pos_size], f64)
        else
            MLIR.IR.TensorType([pos_size, pos_size], f64)
        end
        append!(
            types,
            MLIR.IR.Type[
                scalar_type,
                scalar_type,
                scalar_type,
                scalar_i64_type,
                scalar_type,
                welford_mean_type,
                welford_m2_type,
                scalar_i64_type,
                scalar_i64_type,
                scalar_i64_type,
            ],
        )
    end
    return types
end

function state_operands(
    position,
    gradient,
    potential_energy,
    step_size,
    inverse_mass_matrix,
    adaptation::Union{Nothing,AdaptationState},
    warmup_offset,
    adaptive::Bool,
)
    vals = Any[position, gradient, potential_energy, step_size, inverse_mass_matrix]
    if adaptive
        da = adaptation.dual_averaging
        append!(
            vals,
            Any[
                da.log_step_size,
                da.log_step_size_avg,
                da.gradient_avg,
                da.step_count,
                da.prox_center,
                adaptation.welford.mean,
                adaptation.welford.m2,
                adaptation.welford.n,
                adaptation.window_idx,
                warmup_offset,
            ],
        )
    end
    return MLIR.IR.Value[TracedUtils.get_mlir_data(v) for v in vals]
end

# warmup_offset is not part of MCMCState; chunked warmup tracks it in Julia.
function unpack_state(schema_names, mcmc_op, pos_size, inv_mass_shape, rng_shape, config)
    slot = Dict{String,Int}(n => 4 + i for (i, n) in enumerate(schema_names))
    r(name) = MLIR.IR.result(mcmc_op, slot[name])

    inv_mass = if length(inv_mass_shape) == 1
        TracedRArray{Float64,1}((), r("inverse_mass_matrix"), inv_mass_shape)
    else
        TracedRArray{Float64,2}((), r("inverse_mass_matrix"), inv_mass_shape)
    end

    adaptation = if haskey(slot, "window_idx")
        welford_m2 = if _mass_is_diagonal(inv_mass_shape)
            TracedRArray{Float64,1}((), r("welford_m2"), (pos_size,))
        else
            TracedRArray{Float64,2}((), r("welford_m2"), (pos_size, pos_size))
        end
        da = DualAveragingState(
            TracedRArray{Float64,0}((), r("da_log_step_size"), ()),
            TracedRArray{Float64,0}((), r("da_log_step_size_avg"), ()),
            TracedRArray{Float64,0}((), r("da_gradient_avg"), ()),
            TracedRArray{Int64,0}((), r("da_step_count"), ()),
            TracedRArray{Float64,0}((), r("da_prox_center"), ()),
        )
        welford = WelfordState(
            TracedRArray{Float64,1}((), r("welford_mean"), (pos_size,)),
            welford_m2,
            TracedRArray{Int64,0}((), r("welford_n"), ()),
        )
        AdaptationState(da, welford, TracedRArray{Int64,0}((), r("window_idx"), ()))
    else
        nothing
    end

    return MCMCState(
        TracedRArray{Float64,2}((), r("position"), (1, pos_size)),
        TracedRArray{Float64,2}((), r("gradient"), (1, pos_size)),
        TracedRArray{Float64,0}((), r("potential_energy"), ()),
        TracedRArray{Float64,0}((), r("step_size"), ()),
        inv_mass,
        TracedRArray{UInt64,1}((), MLIR.IR.result(mcmc_op, 4), rng_shape),
        adaptation;
        config=config,
    )
end

function _chain_slice(
    mcmc_op, result_idx::Int, ::Type{T}, single_shape::Tuple, num_chains::Int, c::Int
) where {T}
    nd = length(single_shape) + 1
    cm = TracedRArray{T,nd}(
        (), MLIR.IR.result(mcmc_op, result_idx), (num_chains, single_shape...)
    )
    lo = Int[c, ones(Int, length(single_shape))...]
    hi = Int[c, collect(Int, single_shape)...]
    sl = Reactant.Ops.slice(cm, lo, hi)
    return Reactant.Ops.reshape(sl, collect(Int, single_shape))
end

# output_rng_state stays the last chain's key; each state gets its own chain's key.
function unpack_states_multichain(
    schema_names, mcmc_op, pos_size, inv_mass_shape, rng_shape, config, num_chains
)
    slot = Dict{String,Int}(n => 4 + i for (i, n) in enumerate(schema_names))
    has_adapt = haskey(slot, "window_idx")
    diagonal = _mass_is_diagonal(inv_mass_shape)
    states = MCMCState[]
    for c in 1:num_chains
        pos = _chain_slice(mcmc_op, slot["position"], Float64, (1, pos_size), num_chains, c)
        grad = _chain_slice(
            mcmc_op, slot["gradient"], Float64, (1, pos_size), num_chains, c
        )
        pe = _chain_slice(mcmc_op, slot["potential_energy"], Float64, (), num_chains, c)
        ss = _chain_slice(mcmc_op, slot["step_size"], Float64, (), num_chains, c)
        inv_mass = _chain_slice(
            mcmc_op,
            slot["inverse_mass_matrix"],
            Float64,
            Tuple(inv_mass_shape),
            num_chains,
            c,
        )
        rng_c = _chain_slice(mcmc_op, slot["rng"], UInt64, Tuple(rng_shape), num_chains, c)
        adaptation = if has_adapt
            m2_shape = diagonal ? (pos_size,) : (pos_size, pos_size)
            da = DualAveragingState(
                _chain_slice(mcmc_op, slot["da_log_step_size"], Float64, (), num_chains, c),
                _chain_slice(
                    mcmc_op, slot["da_log_step_size_avg"], Float64, (), num_chains, c
                ),
                _chain_slice(mcmc_op, slot["da_gradient_avg"], Float64, (), num_chains, c),
                _chain_slice(mcmc_op, slot["da_step_count"], Int64, (), num_chains, c),
                _chain_slice(mcmc_op, slot["da_prox_center"], Float64, (), num_chains, c),
            )
            welford = WelfordState(
                _chain_slice(
                    mcmc_op, slot["welford_mean"], Float64, (pos_size,), num_chains, c
                ),
                _chain_slice(mcmc_op, slot["welford_m2"], Float64, m2_shape, num_chains, c),
                _chain_slice(mcmc_op, slot["welford_n"], Int64, (), num_chains, c),
            )
            AdaptationState(
                da,
                welford,
                _chain_slice(mcmc_op, slot["window_idx"], Int64, (), num_chains, c),
            )
        else
            nothing
        end
        push!(
            states, MCMCState(pos, grad, pe, ss, inv_mass, rng_c, adaptation; config=config)
        )
    end
    return states
end

function _stack_chain_major(arrs::Vector)
    reshaped = [Reactant.Ops.reshape(a, Int[1, size(a)...]) for a in arrs]
    return Reactant.Ops.concatenate(reshaped, 1)
end

function state_operands_multichain(states::Vector{MCMCState}, warmup_offset, adaptive::Bool)
    stackf(f) = TracedUtils.get_mlir_data(_stack_chain_major([f(s) for s in states]))
    vals = MLIR.IR.Value[
        stackf(s -> s.position),
        stackf(s -> s.gradient),
        stackf(s -> s.potential_energy),
        stackf(s -> s.step_size),
        stackf(s -> s.inverse_mass_matrix),
    ]
    if adaptive
        append!(
            vals,
            MLIR.IR.Value[
                stackf(s -> s.adaptation.dual_averaging.log_step_size),
                stackf(s -> s.adaptation.dual_averaging.log_step_size_avg),
                stackf(s -> s.adaptation.dual_averaging.gradient_avg),
                stackf(s -> s.adaptation.dual_averaging.step_count),
                stackf(s -> s.adaptation.dual_averaging.prox_center),
                stackf(s -> s.adaptation.welford.mean),
                stackf(s -> s.adaptation.welford.m2),
                stackf(s -> s.adaptation.welford.n),
                stackf(s -> s.adaptation.window_idx),
                TracedUtils.get_mlir_data(
                    _stack_chain_major([warmup_offset for _ in states])
                ),
            ],
        )
    end
    push!(vals, stackf(s -> s.rng))
    return vals
end

# NumPyro defaults to :parallel for num_chains > 1.
const _CHAIN_METHODS = (:sequential, :parallel, :vectorized)
_default_chain_method(num_chains::Int) = num_chains > 1 ? :parallel : :sequential

function _validate_chain_method(chain_method::Symbol)
    chain_method in _CHAIN_METHODS || error(
        "Unknown chain_method: $chain_method. Supported: :sequential, :parallel, :vectorized",
    )
    chain_method == :vectorized && error(
        "chain_method=:vectorized is not implemented yet (planned for H2). Use " *
        ":parallel (default) or :sequential.",
    )
    return chain_method
end

# Bitwise equal to impulse::RandomSplitOp (jax.random.split); all chain methods share it.
function _random_split(rng::ReactantRNG, n::Int)
    seed_val = TracedUtils.get_mlir_data(rng.seed)
    seed_ty = MLIR.IR.type(seed_val)
    op = MLIR.Dialects.impulse.randomSplit(
        seed_val; output_rng_states=[seed_ty for _ in 1:n]
    )
    sz = size(rng.seed)
    return [ReactantRNG(TracedRArray{UInt64,1}((), MLIR.IR.result(op, c), sz)) for c in 1:n]
end

function split_chain_rngs(rng::ReactantRNG, n::Int)
    r = copy(rng)
    split_fn = rr -> Tuple(s.seed for s in _random_split(rr, n))
    compiled = Reactant.Compiler.compile(split_fn, (r,); optimize=:probprog)
    seeds = compiled(r)
    return ReactantRNG[ReactantRNG(copy(s)) for s in seeds]
end

# Safe only because every chain owns distinct concrete inputs (no shared donated buffers).
function _dispatch_parallel(thunk, per_chain_inputs::Vector)
    n = length(per_chain_inputs)
    results = Vector{Any}(undef, n)
    t0 = time()
    tasks = Vector{Task}(undef, n)
    for c in 1:n
        inputs = per_chain_inputs[c]
        tasks[c] = Threads.@spawn thunk(inputs...)
    end
    for c in 1:n
        results[c] = fetch(tasks[c])
    end
    return results, time() - t0
end

function mcmc(
    rng::AbstractRNG,
    original_trace,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    num_chains::Int=1,
    chain_method::Symbol=_default_chain_method(num_chains),
    algorithm::Symbol=:HMC,
    inverse_mass_matrix=nothing,
    step_size=nothing,
    trajectory_length::Float64=2π,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    num_warmup::Int=0,
    num_samples::Int=1,
    thinning::Int=1,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    num_steps::Int=0,
    _layout::Union{Nothing,Base.RefValue}=nothing,
) where {Nargs}
    _validate_chain_method(chain_method)
    # In one traced call, :parallel emits the same unrolled IR as :sequential.
    args = (rng, args...)

    tt = TracedTrace()
    ppf = ScopedValues.with(TRACING_TRACE => tt) do
        return process_probprog_function(f, args, "mcmc")
    end

    (;
        f_name,
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
    ) = ppf

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)
    _layout === nothing || (_layout[] = tt)

    all_addresses_attr = build_selection_attr(tt)

    # Positions use position_layout(selection) order; relayout restores select order.
    layout = position_layout(selection)
    relayout = layout_column_ranges(tt.entries, layout, selection)

    # Samples are constrained; NUTS state is unconstrained (narrower for SIMPLEX).
    selection_pos_size = 0
    unconstrained_pos_size = 0
    for address in layout
        for entry in tt.entries
            entry_path = [entry.parent_path..., entry.symbol]
            if entry_path == collect(address.path)
                selection_pos_size += entry.num_elements
                unconstrained_pos_size += unconstrained_num_elements(entry)
                break
            end
        end
    end

    selection_attr = MLIR.IR.Attribute[]
    for address in layout
        address_attr = MLIR.IR.Attribute[]
        for sym in address.path
            sym_addr = reinterpret(UInt64, pointer_from_objref(sym))
            push!(
                address_attr,
                MLIR.IR.Attribute(
                    MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), sym_addr)
                ),
            )
        end
        push!(selection_attr, MLIR.IR.Attribute(address_attr))
    end

    collection_size = num_samples ÷ thinning
    # Diagnostics columns: [accepted, diverging].
    trace_dims = if num_chains > 1
        [num_chains, collection_size, selection_pos_size]
    else
        [collection_size, selection_pos_size]
    end
    diag_dims =
        num_chains > 1 ? Int64[num_chains, collection_size, 2] : Int64[collection_size, 2]
    ld_dims = num_chains > 1 ? Int64[num_chains, collection_size] : Int64[collection_size]
    trace_type = MLIR.IR.TensorType(trace_dims, MLIR.IR.Type(Float64))
    diagnostics_type = MLIR.IR.TensorType(diag_dims, MLIR.IR.Type(Bool))
    log_densities_type = MLIR.IR.TensorType(ld_dims, MLIR.IR.Type(Float64))

    position_type = MLIR.IR.TensorType([1, unconstrained_pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    hmc_config_attr = nothing
    nuts_config_attr = nothing

    if algorithm == :HMC
        # num_steps > 0 overrides trajectory_length with a fixed leapfrog step count.
        hmc_config_attr = MLIR.IR.Attribute(
            if num_steps > 0
                MLIR.API.enzymeHMCConfigAttrGetWithNumSteps(
                    MLIR.IR.current_context(),
                    trajectory_length,
                    adapt_step_size,
                    adapt_mass_matrix,
                    Int64(num_steps),
                )
            else
                MLIR.API.enzymeHMCConfigAttrGet(
                    MLIR.IR.current_context(),
                    trajectory_length,
                    adapt_step_size,
                    adapt_mass_matrix,
                )
            end,
        )
    elseif algorithm == :NUTS
        nuts_config_attr = MLIR.IR.Attribute(
            MLIR.API.enzymeNUTSConfigAttrGet(
                MLIR.IR.current_context(),
                max_tree_depth,
                true,
                max_delta_energy,
                adapt_step_size,
                adapt_mass_matrix,
            ),
        )
    else
        error("Unknown MCMC algorithm: $algorithm. Supported: :HMC, :NUTS")
    end

    inverse_mass_matrix_val = if isnothing(inverse_mass_matrix)
        nothing
    else
        TracedUtils.get_mlir_data(inverse_mass_matrix)
    end
    step_size_val = isnothing(step_size) ? nothing : TracedUtils.get_mlir_data(step_size)

    inv_mass_type = if isnothing(inverse_mass_matrix_val)
        position_type
    else
        MLIR.IR.type(inverse_mass_matrix_val)
    end

    trace_val = TracedUtils.get_mlir_data(original_trace)

    schema_names = state_schema_names(false, num_chains)

    inv_mass_shape = if isnothing(inverse_mass_matrix)
        (1, unconstrained_pos_size)
    else
        size(inverse_mass_matrix)
    end

    mcmc_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        MLIR.IR.Value[],
        trace_val;
        init_inverse_mass_matrix=inverse_mass_matrix_val,
        init_step_size=step_size_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        log_densities=log_densities_type,
        output_rng_state=mlir_result_types[1],
        state_out=state_out_types(
            false,
            position_type,
            scalar_type,
            inv_mass_type,
            unconstrained_pos_size,
            true;
            num_chains,
            inv_mass_shape,
            rng_shape=size(rng.seed),
        ),
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
        all_addresses=all_addresses_attr,
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
        state_schema=_state_schema_attr(schema_names),
        num_warmup=Int64(num_warmup),
        num_samples=Int64(num_samples),
        thinning=Int64(thinning),
        num_chains=Int64(num_chains),
    )

    traced_result = process_probprog_outputs(
        mcmc_op,
        linear_results,
        traced_result,
        f,
        args,
        fnwrapped,
        resprefix,
        argprefix,
        3,
        true,
    )

    config = MCMCConfig(;
        algorithm, max_tree_depth, max_delta_energy, trajectory_length, thinning
    )

    if num_chains > 1
        new_trace = TracedRArray{Float64,3}(
            (),
            MLIR.IR.result(mcmc_op, 1),
            (num_chains, collection_size, selection_pos_size),
        )
        diagnostics = TracedRArray{Bool,3}(
            (), MLIR.IR.result(mcmc_op, 2), (num_chains, collection_size, 2)
        )
        log_densities = TracedRArray{Float64,2}(
            (), MLIR.IR.result(mcmc_op, 3), (num_chains, collection_size)
        )
        state = unpack_states_multichain(
            schema_names,
            mcmc_op,
            unconstrained_pos_size,
            inv_mass_shape,
            size(rng.seed),
            config,
            num_chains,
        )
        new_trace = _relayout_to_selection(new_trace, relayout)
        return new_trace, diagnostics, log_densities, traced_result, state
    end

    new_trace = TracedRArray{Float64,2}(
        (), MLIR.IR.result(mcmc_op, 1), (collection_size, selection_pos_size)
    )
    diagnostics = TracedRArray{Bool,2}((), MLIR.IR.result(mcmc_op, 2), (collection_size, 2))
    log_densities = TracedRArray{Float64,1}(
        (), MLIR.IR.result(mcmc_op, 3), (collection_size,)
    )

    state = unpack_state(
        schema_names,
        mcmc_op,
        unconstrained_pos_size,
        inv_mass_shape,
        size(rng.seed),
        config,
    )

    new_trace = _relayout_to_selection(new_trace, relayout)
    return new_trace, diagnostics, log_densities, traced_result, state
end

function _infer(
    rng::AbstractRNG,
    logdensity_fn::Function,
    initial_position,
    args::Vararg{Any,Nargs};
    algorithm::Symbol=:NUTS,
    inverse_mass_matrix=nothing,
    step_size=nothing,
    initial_gradient=nothing,
    initial_potential_energy=nothing,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    num_warmup::Int=0,
    num_samples::Int=1,
    thinning::Int=1,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    trajectory_length::Float64=2π,
    strong_zero::Bool=false,
    total_warmup::Union{Nothing,Int}=nothing,
    warmup_offset=nothing,
    adaptation_state::Union{Nothing,AdaptationState}=nothing,
) where {Nargs}
    pos_size = length(initial_position)

    sample_pos = TracedRArray{Float64,1}((), nothing, (pos_size,))
    logpdf_fn_attr, extra_mlir_data = trace_logpdf_function(logdensity_fn, sample_pos, args)

    initial_position_2d = if ndims(initial_position) == 1
        reshape(initial_position, 1, pos_size)
    else
        initial_position
    end
    initial_gradient_2d = if !isnothing(initial_gradient) && ndims(initial_gradient) == 1
        reshape(initial_gradient, 1, pos_size)
    else
        initial_gradient
    end

    rng_args = (rng,)
    ppf = process_probprog_function(identity, rng_args, "mcmc_logpdf")
    (;
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
    ) = ppf

    append!(mlir_caller_args, extra_mlir_data)

    hmc_config_attr = nothing
    nuts_config_attr = nothing

    if algorithm == :HMC
        hmc_config_attr = MLIR.API.enzymeHMCConfigAttrGet(
            MLIR.IR.current_context(), trajectory_length, adapt_step_size, adapt_mass_matrix
        )
    elseif algorithm == :NUTS
        nuts_config_attr = MLIR.API.enzymeNUTSConfigAttrGet(
            MLIR.IR.current_context(),
            max_tree_depth,
            true,
            max_delta_energy,
            adapt_step_size,
            adapt_mass_matrix,
        )
    else
        error("Unknown MCMC algorithm: $algorithm. Supported: :HMC, :NUTS")
    end

    inverse_mass_matrix_val = if isnothing(inverse_mass_matrix)
        nothing
    else
        TracedUtils.get_mlir_data(inverse_mass_matrix)
    end
    step_size_val = isnothing(step_size) ? nothing : TracedUtils.get_mlir_data(step_size)
    initial_position_val = TracedUtils.get_mlir_data(initial_position_2d)

    collection_size = num_samples ÷ thinning
    trace_type = MLIR.IR.TensorType([collection_size, pos_size], MLIR.IR.Type(Float64))
    diagnostics_type = MLIR.IR.TensorType(
        [Int64(collection_size), Int64(2)], MLIR.IR.Type(Bool)
    )
    log_densities_type = MLIR.IR.TensorType([Int64(collection_size)], MLIR.IR.Type(Float64))

    position_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    inv_mass_type = if isnothing(inverse_mass_matrix_val)
        position_type
    else
        MLIR.IR.type(inverse_mass_matrix_val)
    end

    inv_mass_shape =
        isnothing(inverse_mass_matrix) ? (1, pos_size) : size(inverse_mass_matrix)
    diagonal = _mass_is_diagonal(inv_mass_shape)

    # Keep adaptation when resuming so the cached sample kernels see a stable state.
    adaptive = adapt_step_size || adapt_mass_matrix || !isnothing(adaptation_state)
    resume =
        !isnothing(initial_gradient_2d) &&
        !isnothing(initial_potential_energy) &&
        (!adaptive || !isnothing(adaptation_state))

    schema_names = state_schema_names(adaptive)

    state_in_vals = if resume
        offset = if (adaptive && isnothing(warmup_offset))
            Ops.constant(fill(Int64(0)))
        else
            warmup_offset
        end
        state_operands(
            initial_position_2d,
            initial_gradient_2d,
            initial_potential_energy,
            step_size,
            inverse_mass_matrix,
            adaptation_state,
            offset,
            adaptive,
        )
    else
        MLIR.IR.Value[]
    end

    total_warmup_attr = isnothing(total_warmup) ? nothing : Int64(total_warmup)

    mcmc_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        state_in_vals;
        init_inverse_mass_matrix=inverse_mass_matrix_val,
        init_step_size=step_size_val,
        init_position=initial_position_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        log_densities=log_densities_type,
        output_rng_state=mlir_result_types[1],
        state_out=state_out_types(
            adaptive, position_type, scalar_type, inv_mass_type, pos_size, diagonal
        ),
        total_warmup=total_warmup_attr,
        logpdf_fn=logpdf_fn_attr,
        selection=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        all_addresses=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
        state_schema=_state_schema_attr(schema_names),
        num_warmup=Int64(num_warmup),
        num_samples=Int64(num_samples),
        thinning=Int64(thinning),
        autodiff_attrs=MLIR.IR.Attribute(
            Dict("strong_zero" => MLIR.IR.Attribute(strong_zero))
        ),
    )

    traced_result = process_probprog_outputs(
        mcmc_op,
        linear_results,
        traced_result,
        identity,
        rng_args,
        fnwrapped,
        resprefix,
        argprefix,
        3,
        true,
    )

    new_trace = TracedRArray{Float64,2}(
        (), MLIR.IR.result(mcmc_op, 1), (collection_size, pos_size)
    )
    diagnostics = TracedRArray{Bool,2}((), MLIR.IR.result(mcmc_op, 2), (collection_size, 2))
    log_densities = TracedRArray{Float64,1}(
        (), MLIR.IR.result(mcmc_op, 3), (collection_size,)
    )

    state = unpack_state(
        schema_names,
        mcmc_op,
        pos_size,
        inv_mass_shape,
        size(rng.seed),
        MCMCConfig(;
            algorithm,
            max_tree_depth,
            max_delta_energy,
            trajectory_length,
            thinning,
            strong_zero,
        ),
    )

    return new_trace, diagnostics, log_densities, traced_result, state
end

function _infer(
    state::MCMCState,
    logdensity_fn::Function,
    args::Vararg{Any,Nargs};
    inverse_mass_matrix=state.inverse_mass_matrix,
    step_size=state.step_size,
    initial_gradient=state.gradient,
    initial_potential_energy=state.potential_energy,
    kwargs...,
) where {Nargs}
    return _infer(
        ReactantRNG(state.rng),
        logdensity_fn,
        state.position,
        args...;
        inverse_mass_matrix,
        step_size,
        initial_gradient,
        initial_potential_energy,
        adaptation_state=state.adaptation,
        kwargs...,
    )
end

# initial_position must be chain-major (leading dim == num_chains), as in NumPyro.
function _infer_multichain(
    rng::AbstractRNG,
    logdensity_fn::Function,
    initial_position,
    args::Vararg{Any,Nargs};
    num_chains::Int,
    algorithm::Symbol=:NUTS,
    inverse_mass_matrix=nothing,
    step_size=nothing,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    num_warmup::Int=0,
    num_samples::Int=1,
    thinning::Int=1,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    trajectory_length::Float64=2π,
    strong_zero::Bool=false,
    total_warmup::Union{Nothing,Int}=nothing,
    warmup_offset=nothing,
    resume_states::Union{Nothing,Vector{MCMCState}}=nothing,
) where {Nargs}
    resume = resume_states !== nothing

    initial_position_3d = if resume
        _stack_chain_major([s.position for s in resume_states])
    elseif ndims(initial_position) == 2
        reshape(initial_position, num_chains, 1, :)
    else
        initial_position
    end
    if size(initial_position_3d, 1) != num_chains
        error(
            "multi-chain `initial_position` leading dimension " *
            "($(size(initial_position_3d, 1))) must equal num_chains ($num_chains)",
        )
    end
    pos_size = size(initial_position_3d, 3)

    sample_pos = TracedRArray{Float64,1}((), nothing, (pos_size,))
    logpdf_fn_attr, extra_mlir_data = trace_logpdf_function(logdensity_fn, sample_pos, args)

    rng_args = (rng,)
    ppf = process_probprog_function(identity, rng_args, "mcmc_logpdf")
    (;
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
    ) = ppf
    append!(mlir_caller_args, extra_mlir_data)

    hmc_config_attr = nothing
    nuts_config_attr = nothing
    if algorithm == :HMC
        hmc_config_attr = MLIR.API.enzymeHMCConfigAttrGet(
            MLIR.IR.current_context(), trajectory_length, adapt_step_size, adapt_mass_matrix
        )
    elseif algorithm == :NUTS
        nuts_config_attr = MLIR.API.enzymeNUTSConfigAttrGet(
            MLIR.IR.current_context(),
            max_tree_depth,
            true,
            max_delta_energy,
            adapt_step_size,
            adapt_mass_matrix,
        )
    else
        error("Unknown MCMC algorithm: $algorithm. Supported: :HMC, :NUTS")
    end

    imm_ref = resume ? resume_states[1].inverse_mass_matrix : inverse_mass_matrix
    ss_ref = resume ? resume_states[1].step_size : step_size
    inverse_mass_matrix_val =
        isnothing(imm_ref) ? nothing : TracedUtils.get_mlir_data(imm_ref)
    step_size_val = isnothing(ss_ref) ? nothing : TracedUtils.get_mlir_data(ss_ref)
    initial_position_val = TracedUtils.get_mlir_data(initial_position_3d)

    collection_size = num_samples ÷ thinning
    trace_type = MLIR.IR.TensorType(
        [num_chains, collection_size, pos_size], MLIR.IR.Type(Float64)
    )
    # Diagnostics columns: [accepted, diverging].
    diagnostics_type = MLIR.IR.TensorType(
        [num_chains, collection_size, 2], MLIR.IR.Type(Bool)
    )
    log_densities_type = MLIR.IR.TensorType(
        [num_chains, collection_size], MLIR.IR.Type(Float64)
    )

    position_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))
    inv_mass_type = if isnothing(inverse_mass_matrix_val)
        position_type
    else
        MLIR.IR.type(inverse_mass_matrix_val)
    end
    inv_mass_shape = if resume
        size(resume_states[1].inverse_mass_matrix)
    elseif isnothing(inverse_mass_matrix)
        (1, pos_size)
    else
        size(inverse_mass_matrix)
    end
    diagonal = _mass_is_diagonal(inv_mass_shape)

    adaptive =
        adapt_step_size ||
        adapt_mass_matrix ||
        (resume && resume_states[1].adaptation !== nothing)
    schema_names = state_schema_names(adaptive, num_chains)

    state_in_vals = if resume
        offset = if adaptive && isnothing(warmup_offset)
            Ops.constant(fill(Int64(0)))
        else
            warmup_offset
        end
        state_operands_multichain(resume_states, offset, adaptive)
    else
        MLIR.IR.Value[]
    end

    total_warmup_attr = isnothing(total_warmup) ? nothing : Int64(total_warmup)

    mcmc_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        state_in_vals;
        init_inverse_mass_matrix=inverse_mass_matrix_val,
        init_step_size=step_size_val,
        init_position=initial_position_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        log_densities=log_densities_type,
        output_rng_state=mlir_result_types[1],
        state_out=state_out_types(
            adaptive,
            position_type,
            scalar_type,
            inv_mass_type,
            pos_size,
            diagonal;
            num_chains,
            inv_mass_shape,
            rng_shape=size(rng.seed),
        ),
        total_warmup=total_warmup_attr,
        logpdf_fn=logpdf_fn_attr,
        selection=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        all_addresses=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
        state_schema=_state_schema_attr(schema_names),
        num_warmup=Int64(num_warmup),
        num_samples=Int64(num_samples),
        thinning=Int64(thinning),
        num_chains=Int64(num_chains),
        autodiff_attrs=MLIR.IR.Attribute(
            Dict("strong_zero" => MLIR.IR.Attribute(strong_zero))
        ),
    )

    traced_result = process_probprog_outputs(
        mcmc_op,
        linear_results,
        traced_result,
        identity,
        rng_args,
        fnwrapped,
        resprefix,
        argprefix,
        3,
        true,
    )

    new_trace = TracedRArray{Float64,3}(
        (), MLIR.IR.result(mcmc_op, 1), (num_chains, collection_size, pos_size)
    )
    diagnostics = TracedRArray{Bool,3}(
        (), MLIR.IR.result(mcmc_op, 2), (num_chains, collection_size, 2)
    )
    log_densities = TracedRArray{Float64,2}(
        (), MLIR.IR.result(mcmc_op, 3), (num_chains, collection_size)
    )
    states = unpack_states_multichain(
        schema_names,
        mcmc_op,
        pos_size,
        inv_mass_shape,
        size(rng.seed),
        MCMCConfig(;
            algorithm,
            max_tree_depth,
            max_delta_energy,
            trajectory_length,
            thinning,
            strong_zero,
        ),
        num_chains,
    )
    return new_trace, diagnostics, log_densities, traced_result, states
end

function mcmc_logpdf(
    rng::AbstractRNG,
    logdensity_fn::Function,
    initial_position,
    args::Vararg{Any,Nargs};
    algorithm::Symbol=:NUTS,
    inverse_mass_matrix=nothing,
    step_size=nothing,
    initial_gradient=nothing,
    initial_potential_energy=nothing,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    num_warmup::Int=0,
    num_samples::Int=1,
    thinning::Int=1,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    trajectory_length::Float64=2π,
    strong_zero::Bool=false,
    num_chains::Int=1,
    chain_method::Symbol=_default_chain_method(num_chains),
) where {Nargs}
    _validate_chain_method(chain_method)
    if num_chains > 1
        return _infer_multichain(
            rng,
            logdensity_fn,
            initial_position,
            args...;
            num_chains,
            algorithm,
            inverse_mass_matrix,
            step_size,
            max_tree_depth,
            max_delta_energy,
            num_warmup,
            num_samples,
            thinning,
            adapt_step_size,
            adapt_mass_matrix,
            trajectory_length,
            strong_zero,
        )
    end
    return _infer(
        rng,
        logdensity_fn,
        initial_position,
        args...;
        algorithm,
        inverse_mass_matrix,
        step_size,
        initial_gradient,
        initial_potential_energy,
        max_tree_depth,
        max_delta_energy,
        num_warmup,
        num_samples,
        thinning,
        adapt_step_size,
        adapt_mass_matrix,
        trajectory_length,
        strong_zero,
    )
end

function mcmc(
    state::MCMCState,
    original_trace,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    inverse_mass_matrix=state.inverse_mass_matrix,
    step_size=state.step_size,
    num_warmup::Int=0,
    adapt_step_size::Bool=false,
    adapt_mass_matrix::Bool=false,
    kwargs...,
) where {Nargs}
    return mcmc(
        ReactantRNG(state.rng),
        original_trace,
        f,
        args...;
        selection,
        inverse_mass_matrix,
        step_size,
        num_warmup,
        adapt_step_size,
        adapt_mass_matrix,
        kwargs...,
    )
end

function mcmc_logpdf(
    state::MCMCState,
    logdensity_fn::Function,
    args::Vararg{Any,Nargs};
    inverse_mass_matrix=state.inverse_mass_matrix,
    step_size=state.step_size,
    initial_gradient=state.gradient,
    initial_potential_energy=state.potential_energy,
    algorithm::Symbol=state.config.algorithm,
    max_tree_depth::Int=state.config.max_tree_depth,
    max_delta_energy::Float64=state.config.max_delta_energy,
    trajectory_length::Float64=state.config.trajectory_length,
    thinning::Int=state.config.thinning,
    strong_zero::Bool=state.config.strong_zero,
    num_warmup::Int=0,
    num_samples::Int=1,
    adapt_step_size::Bool=false,
    adapt_mass_matrix::Bool=false,
) where {Nargs}
    return _infer(
        state,
        logdensity_fn,
        args...;
        inverse_mass_matrix,
        step_size,
        initial_gradient,
        initial_potential_energy,
        algorithm,
        max_tree_depth,
        max_delta_energy,
        trajectory_length,
        thinning,
        strong_zero,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )
end

function mcmc_logpdf(
    states::Vector{MCMCState},
    logdensity_fn::Function,
    args::Vararg{Any,Nargs};
    algorithm::Symbol=states[1].config.algorithm,
    max_tree_depth::Int=states[1].config.max_tree_depth,
    max_delta_energy::Float64=states[1].config.max_delta_energy,
    trajectory_length::Float64=states[1].config.trajectory_length,
    thinning::Int=states[1].config.thinning,
    strong_zero::Bool=states[1].config.strong_zero,
    num_warmup::Int=0,
    num_samples::Int=1,
    adapt_step_size::Bool=false,
    adapt_mass_matrix::Bool=false,
    total_warmup::Union{Nothing,Int}=nothing,
    warmup_offset=nothing,
) where {Nargs}
    return _infer_multichain(
        ReactantRNG(states[1].rng),
        logdensity_fn,
        nothing,
        args...;
        num_chains=length(states),
        resume_states=states,
        algorithm,
        max_tree_depth,
        max_delta_energy,
        trajectory_length,
        thinning,
        strong_zero,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
        total_warmup,
        warmup_offset,
    )
end

function _format_stats(; step_size=nothing, acc_rate=nothing)
    parts = String[]
    step_size !== nothing && push!(parts, "step_size=$(round(step_size; sigdigits=3))")
    acc_rate !== nothing && push!(parts, "acc. rate=$(round(acc_rate; digits=2))")
    return isempty(parts) ? "" : " | " * join(parts, ", ")
end

function _print_progress(current, total; width=40, stats="", label="Sampling")
    frac = total == 0 ? 1.0 : current / total
    filled = round(Int, frac * width)
    bar = "━"^filled * " "^(width - filled)
    print("\r  $(label) $(bar) $(current)/$(total)$(stats)")
    current == total && println()
    return flush(stdout)
end

function _show_progress(info)
    if info.phase === :warmup
        _print_progress(
            info.step,
            info.total;
            label="Warmup",
            stats=_format_stats(; step_size=info.step_size),
        )
    else
        _print_progress(
            info.step,
            info.total;
            label="Sampling",
            stats=_format_stats(; step_size=info.step_size, acc_rate=info.acceptance_rate),
        )
    end
    return nothing
end

function save_state(filename::String, state::MCMCState)
    data = Dict{String,Any}("sampler" => _sampler_to_dict(state))
    if !isnothing(state.adaptation)
        data["adaptation"] = _adaptation_to_dict(state.adaptation)
    end
    return open(io -> Serialization.serialize(io, data), filename, "w")
end

function load_state(filename::String)
    data = open(Serialization.deserialize, filename)
    state = _sampler_from_dict(data["sampler"])
    if haskey(data, "adaptation")
        state.adaptation = _adaptation_from_dict(data["adaptation"])
    end
    return state
end

function _warmup_info(st::MCMCState, step, total)
    return (;
        phase=:warmup,
        step=step,
        total=total,
        step_size=Float64(Array(st.step_size)[]),
        inverse_mass_matrix=Array(st.inverse_mass_matrix),
        acceptance_rate=nothing,
        samples=nothing,
        state=st,
    )
end

function _compile_warmup_kernel(
    state::MCMCState,
    logpdf_fn,
    num_steps::Int,
    total::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    mcmc_kwargs,
)
    fn = function (st_in::MCMCState, lf, off)
        _, _, _, _, st_out = _infer(
            st_in,
            lf;
            num_warmup=num_steps,
            num_samples=0,
            adapt_step_size,
            adapt_mass_matrix,
            total_warmup=total,
            warmup_offset=off,
            mcmc_kwargs...,
        )
        return st_out
    end
    return Reactant.Compiler.compile(
        fn, (state, logpdf_fn, ConcreteRNumber(Int64(0))); optimize=:probprog
    )
end

function _compile_sample_kernel(state::MCMCState, logpdf_fn, num_samples::Int, mcmc_kwargs)
    fn = function (st::MCMCState, lf)
        samples, diag, _, _, st_out = _infer(
            st,
            lf;
            num_warmup=0,
            num_samples=num_samples,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            mcmc_kwargs...,
        )
        return samples, diag, st_out
    end
    return Reactant.Compiler.compile(fn, (state, logpdf_fn); optimize=:probprog)
end

function _run_chain_chunked(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix;
    num_warmup,
    num_samples,
    chunk_size,
    warmup_chunk_size,
    adapt_step_size,
    adapt_mass_matrix,
    initial_gradient,
    initial_potential_energy,
    progress_bar,
    callback,
    mcmc_kwargs,
)
    pos_size = length(initial_position)
    all_chunks = Matrix{Float64}[]
    collected = 0

    state = MCMCState(
        initial_position,
        initial_gradient,
        initial_potential_energy,
        step_size,
        inverse_mass_matrix,
        rng.seed;
        config=MCMCConfig(; mcmc_kwargs...),
    )

    if num_warmup > 0
        done = 0
        kernel = nothing
        kernel_key = nothing
        while done < num_warmup
            nsteps = min(warmup_chunk_size, num_warmup - done)
            key = (nsteps, isnothing(state.gradient))
            if key != kernel_key
                kernel = _compile_warmup_kernel(
                    state,
                    logpdf_fn,
                    nsteps,
                    num_warmup,
                    adapt_step_size,
                    adapt_mass_matrix,
                    mcmc_kwargs,
                )
                kernel_key = key
            end
            state = kernel(state, logpdf_fn, ConcreteRNumber(Int64(done)))
            done += nsteps
            info = _warmup_info(state, done, num_warmup)
            progress_bar && _show_progress(info)
            callback === nothing || callback(info)
        end
    end

    kernel = nothing
    kernel_key = nothing
    while collected < num_samples
        nsamp = min(chunk_size, num_samples - collected)
        key = (nsamp, isnothing(state.gradient))
        if key != kernel_key
            kernel = _compile_sample_kernel(state, logpdf_fn, nsamp, mcmc_kwargs)
            kernel_key = key
        end
        samples, diag, state = kernel(state, logpdf_fn)
        chunk = Array(samples)
        diag_arr = Array(diag)
        push!(all_chunks, chunk)
        collected += nsamp
        sampling_info = (;
            phase=:sampling,
            step=collected,
            total=num_samples,
            step_size=Float64(Array(state.step_size)[]),
            inverse_mass_matrix=Array(state.inverse_mass_matrix),
            acceptance_rate=(
                isempty(diag_arr) ? nothing : sum(view(diag_arr, :, 1)) / size(diag_arr, 1)
            ),
            samples=chunk,
            state=state,
        )
        progress_bar && _show_progress(sampling_info)
        callback === nothing || callback(sampling_info)
    end

    result = isempty(all_chunks) ? Matrix{Float64}(undef, 0, pos_size) : vcat(all_chunks...)
    return result, state
end

# Chain c uses split(rng, num_chains)[c] and row c of initial_position, as :sequential.
function _run_chain_parallel(
    rng,
    logpdf_fn,
    initial_position;
    num_chains::Int,
    num_warmup::Int,
    num_samples::Int,
    step_size,
    inverse_mass_matrix,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    mcmc_kwargs,
)
    keys = split_chain_rngs(rng, num_chains)
    pos_host = Array(initial_position)
    size(pos_host, 1) == num_chains || error(
        "multi-chain `initial_position` leading dimension ($(size(pos_host, 1))) " *
        "must equal num_chains ($num_chains)",
    )

    single_fn = function (r, logpdf_fn, pos, ss, imm)
        s, _, _, _, st = mcmc_logpdf(
            r,
            logpdf_fn,
            pos;
            num_chains=1,
            step_size=ss,
            inverse_mass_matrix=imm,
            num_warmup,
            num_samples,
            adapt_step_size,
            adapt_mass_matrix,
            mcmc_kwargs...,
        )
        return s, st
    end

    chain_inputs =
        c -> (
            keys[c],
            logpdf_fn,
            Reactant.to_rarray(pos_host[c, :]),
            (step_size === nothing ? nothing : copy(step_size)),
            (inverse_mass_matrix === nothing ? nothing : copy(inverse_mass_matrix)),
        )

    compiled = Reactant.Compiler.compile(single_fn, chain_inputs(1); optimize=:probprog)

    per_chain_inputs = Any[chain_inputs(c) for c in 1:num_chains]
    results, elapsed = _dispatch_parallel(compiled, per_chain_inputs)
    @debug "run_chain(:parallel) dispatched $num_chains chains over $(Threads.nthreads()) threads in $(round(elapsed; digits=3))s"

    samples = cat((reshape(Array(r[1]), 1, size(r[1])...) for r in results)...; dims=1)
    states = MCMCState[r[2] for r in results]
    return samples, states
end

# TODO(#2619): add trace-based mode support
function run_chain(
    rng,
    logpdf_fn::Function,
    initial_position;
    algorithm::Symbol=:NUTS,
    num_warmup::Int=0,
    num_samples::Int=1000,
    chunk_size::Int=100,
    num_chains::Int=1,
    chain_method::Symbol=_default_chain_method(num_chains),
    step_size=nothing,
    inverse_mass_matrix=nothing,
    initial_gradient=nothing,
    initial_potential_energy=nothing,
    progress_bar::Bool=true,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    thinning::Int=1,
    trajectory_length::Float64=2π,
    strong_zero::Bool=false,
    callback=nothing,
    warmup_chunk_size::Int=chunk_size,
)
    _validate_chain_method(chain_method)
    mcmc_kwargs = (;
        algorithm,
        max_tree_depth,
        max_delta_energy,
        trajectory_length,
        thinning,
        strong_zero,
    )

    if num_chains > 1
        if chain_method == :parallel
            return _run_chain_parallel(
                rng,
                logpdf_fn,
                initial_position;
                num_chains,
                num_warmup,
                num_samples,
                step_size,
                inverse_mass_matrix,
                adapt_step_size,
                adapt_mass_matrix,
                mcmc_kwargs,
            )
        end
        monolithic_fn = function (rng, logpdf_fn, pos, ss, imm)
            s, _, _, _, sts = mcmc_logpdf(
                rng,
                logpdf_fn,
                pos;
                num_chains,
                chain_method=:sequential,
                step_size=ss,
                inverse_mass_matrix=imm,
                num_warmup,
                num_samples,
                adapt_step_size,
                adapt_mass_matrix,
                mcmc_kwargs...,
            )
            return s, sts
        end
        args = (rng, logpdf_fn, initial_position, step_size, inverse_mass_matrix)
        compiled = Reactant.Compiler.compile(monolithic_fn, args; optimize=:probprog)
        samples, states = compiled(args...)
        return Array(samples), states
    end

    if !progress_bar && isnothing(callback) && num_samples > 0
        monolithic_fn = function (rng, logpdf_fn, pos, grad, pe, ss, imm)
            s, _, _, _, st = mcmc_logpdf(
                rng,
                logpdf_fn,
                pos;
                step_size=ss,
                inverse_mass_matrix=imm,
                initial_gradient=grad,
                initial_potential_energy=pe,
                num_warmup,
                num_samples,
                adapt_step_size,
                adapt_mass_matrix,
                mcmc_kwargs...,
            )
            return s, st
        end
        args = (
            rng,
            logpdf_fn,
            initial_position,
            initial_gradient,
            initial_potential_energy,
            step_size,
            inverse_mass_matrix,
        )
        compiled = Reactant.Compiler.compile(monolithic_fn, args; optimize=:probprog)
        samples, state = compiled(args...)
        return Array(samples), state
    end

    return _run_chain_chunked(
        rng,
        logpdf_fn,
        initial_position,
        step_size,
        inverse_mass_matrix;
        num_warmup,
        num_samples,
        chunk_size,
        warmup_chunk_size,
        adapt_step_size,
        adapt_mass_matrix,
        initial_gradient,
        initial_potential_energy,
        progress_bar,
        callback,
        mcmc_kwargs,
    )
end

function run_chain(
    state::MCMCState,
    logpdf_fn::Function;
    algorithm::Symbol=state.config.algorithm,
    num_samples::Int=1000,
    chunk_size::Int=100,
    progress_bar::Bool=true,
    max_tree_depth::Int=state.config.max_tree_depth,
    max_delta_energy::Float64=state.config.max_delta_energy,
    thinning::Int=state.config.thinning,
    trajectory_length::Float64=state.config.trajectory_length,
    strong_zero::Bool=state.config.strong_zero,
    callback=nothing,
)
    s = copy(state)
    return run_chain(
        ReactantRNG(s.rng),
        logpdf_fn,
        s.position;
        algorithm,
        num_warmup=0,
        num_samples,
        chunk_size,
        step_size=s.step_size,
        inverse_mass_matrix=s.inverse_mass_matrix,
        initial_gradient=s.gradient,
        initial_potential_energy=s.potential_energy,
        progress_bar,
        max_tree_depth,
        max_delta_energy,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        thinning,
        trajectory_length,
        strong_zero,
        callback,
    )
end
