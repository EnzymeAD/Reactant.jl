import ..Reactant
using ..Reactant: TracedRArray, ReactantRNG, ConcreteRNumber
using Serialization: Serialization

function mcmc(
    rng::AbstractRNG,
    original_trace,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
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
) where {Nargs}
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

    all_addresses_attr = build_selection_attr(tt)

    selection_pos_size = 0
    for address in selection
        for entry in tt.entries
            entry_path = [entry.parent_path..., entry.symbol]
            if entry_path == collect(address.path)
                selection_pos_size += entry.num_elements
                break
            end
        end
    end

    selection_attr = MLIR.IR.Attribute[]
    for address in selection
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
    trace_type = MLIR.IR.TensorType(
        [collection_size, selection_pos_size], MLIR.IR.Type(Float64)
    )
    diagnostics_type = MLIR.IR.TensorType([Int64(collection_size)], MLIR.IR.Type(Bool))

    position_type = MLIR.IR.TensorType([1, selection_pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    hmc_config_attr = nothing
    nuts_config_attr = nothing

    if algorithm == :HMC
        hmc_config_attr = MLIR.IR.Attribute(
            MLIR.API.enzymeHMCConfigAttrGet(
                MLIR.IR.current_context(),
                trajectory_length,
                adapt_step_size,
                adapt_mass_matrix,
            ),
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

    mcmc_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        MLIR.IR.Value[],
        trace_val;
        inverse_mass_matrix=inverse_mass_matrix_val,
        step_size=step_size_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        output_rng_state=mlir_result_types[1],
        final_position=position_type,
        final_gradient=position_type,
        final_potential_energy=scalar_type,
        final_step_size=scalar_type,
        final_inverse_mass_matrix=inv_mass_type,
        adaptation_state_out=MLIR.IR.Type[],
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
        all_addresses=all_addresses_attr,
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
        num_warmup=Int64(num_warmup),
        num_samples=Int64(num_samples),
        thinning=Int64(thinning),
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
        2,
        true,
    )

    new_trace = TracedRArray{Float64,2}(
        (), MLIR.IR.result(mcmc_op, 1), (collection_size, selection_pos_size)
    )
    diagnostics = TracedRArray{Bool,1}((), MLIR.IR.result(mcmc_op, 2), (collection_size,))

    inv_mass_shape =
        isnothing(inverse_mass_matrix) ? (1, selection_pos_size) : size(inverse_mass_matrix)

    state = MCMCState(
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 4), (1, selection_pos_size)),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 5), (1, selection_pos_size)),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 6), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 7), ()),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 8), inv_mass_shape),
        TracedRArray{UInt64,1}((), MLIR.IR.result(mcmc_op, 3), size(rng.seed));
        config=MCMCConfig(;
            algorithm, max_tree_depth, max_delta_energy, trajectory_length, thinning
        ),
    )

    return new_trace, diagnostics, traced_result, state
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
    initial_gradient_val = if isnothing(initial_gradient_2d)
        nothing
    else
        TracedUtils.get_mlir_data(initial_gradient_2d)
    end
    initial_pe_val = if isnothing(initial_potential_energy)
        nothing
    else
        TracedUtils.get_mlir_data(initial_potential_energy)
    end

    collection_size = num_samples ÷ thinning
    trace_type = MLIR.IR.TensorType([collection_size, pos_size], MLIR.IR.Type(Float64))
    diagnostics_type = MLIR.IR.TensorType([Int64(collection_size)], MLIR.IR.Type(Bool))

    position_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    inv_mass_type = if isnothing(inverse_mass_matrix_val)
        position_type
    else
        MLIR.IR.type(inverse_mass_matrix_val)
    end

    scalar_i64_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Int64))
    diagonal = isnothing(inverse_mass_matrix) ? true : (ndims(inverse_mass_matrix) == 1)
    welford_mean_type = MLIR.IR.TensorType([pos_size], MLIR.IR.Type(Float64))
    welford_m2_type = if diagonal
        MLIR.IR.TensorType([pos_size], MLIR.IR.Type(Float64))
    else
        MLIR.IR.TensorType([pos_size, pos_size], MLIR.IR.Type(Float64))
    end

    adaptation_in_vals = if isnothing(adaptation_state)
        MLIR.IR.Value[]
    else
        MLIR.IR.Value[
            TracedUtils.get_mlir_data(x) for x in adaptation_operands(adaptation_state)
        ]
    end
    warmup_offset_val =
        isnothing(warmup_offset) ? nothing : TracedUtils.get_mlir_data(warmup_offset)
    total_warmup_attr = isnothing(total_warmup) ? nothing : Int64(total_warmup)
    adaptation_out_types = MLIR.IR.Type[
        scalar_type,
        scalar_type,
        scalar_type,
        scalar_i64_type,
        scalar_type,
        welford_mean_type,
        welford_m2_type,
        scalar_i64_type,
        scalar_i64_type,
    ]

    mcmc_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        adaptation_in_vals;
        inverse_mass_matrix=inverse_mass_matrix_val,
        step_size=step_size_val,
        initial_position=initial_position_val,
        initial_gradient=initial_gradient_val,
        initial_potential_energy=initial_pe_val,
        warmup_offset=warmup_offset_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        output_rng_state=mlir_result_types[1],
        final_position=position_type,
        final_gradient=position_type,
        final_potential_energy=scalar_type,
        final_step_size=scalar_type,
        final_inverse_mass_matrix=inv_mass_type,
        adaptation_state_out=adaptation_out_types,
        total_warmup=total_warmup_attr,
        logpdf_fn=logpdf_fn_attr,
        selection=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        all_addresses=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
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
        2,
        true,
    )

    new_trace = TracedRArray{Float64,2}(
        (), MLIR.IR.result(mcmc_op, 1), (collection_size, pos_size)
    )
    diagnostics = TracedRArray{Bool,1}((), MLIR.IR.result(mcmc_op, 2), (collection_size,))

    inv_mass_shape =
        isnothing(inverse_mass_matrix) ? (1, pos_size) : size(inverse_mass_matrix)
    inv_mass_ndims = length(inv_mass_shape)

    inv_mass_traced = if inv_mass_ndims == 1
        TracedRArray{Float64,1}((), MLIR.IR.result(mcmc_op, 8), inv_mass_shape)
    else
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 8), inv_mass_shape)
    end

    da = DualAveragingState(
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 9), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 10), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 11), ()),
        TracedRArray{Int64,0}((), MLIR.IR.result(mcmc_op, 12), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 13), ()),
    )
    welford_m2 = if diagonal
        TracedRArray{Float64,1}((), MLIR.IR.result(mcmc_op, 15), (pos_size,))
    else
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 15), (pos_size, pos_size))
    end
    welford = WelfordState(
        TracedRArray{Float64,1}((), MLIR.IR.result(mcmc_op, 14), (pos_size,)),
        welford_m2,
        TracedRArray{Int64,0}((), MLIR.IR.result(mcmc_op, 16), ()),
    )
    window_idx = TracedRArray{Int64,0}((), MLIR.IR.result(mcmc_op, 17), ())
    adaptation = AdaptationState(da, welford, window_idx)

    state = MCMCState(
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 4), (1, pos_size)),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 5), (1, pos_size)),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 6), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 7), ()),
        inv_mass_traced,
        TracedRArray{UInt64,1}((), MLIR.IR.result(mcmc_op, 3), size(rng.seed)),
        adaptation;
        config=MCMCConfig(;
            algorithm,
            max_tree_depth,
            max_delta_energy,
            trajectory_length,
            thinning,
            strong_zero,
        ),
    )

    return new_trace, diagnostics, traced_result, state
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
) where {Nargs}
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
        _, _, _, st_out = _infer(
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
        samples, diag, _, st_out = _infer(
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

function _seed_state(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix;
    adapt_step_size,
    adapt_mass_matrix,
    total_warmup,
    mcmc_kwargs,
)
    seed_fn = function (sr, lf, pos, ss, imm)
        _, _, _, st = _infer(
            ReactantRNG(sr),
            lf,
            pos;
            step_size=ss,
            inverse_mass_matrix=imm,
            num_warmup=0,
            num_samples=0,
            adapt_step_size,
            adapt_mass_matrix,
            total_warmup,
            mcmc_kwargs...,
        )
        return st
    end
    args = (rng.seed, logpdf_fn, initial_position, step_size, inverse_mass_matrix)
    compiled = Reactant.Compiler.compile(seed_fn, args; optimize=:probprog)
    return compiled(args...)
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

    state = nothing

    if num_warmup > 0
        state = _seed_state(
            rng,
            logpdf_fn,
            initial_position,
            step_size,
            inverse_mass_matrix;
            adapt_step_size,
            adapt_mass_matrix,
            total_warmup=num_warmup,
            mcmc_kwargs,
        )
        done = 0
        kernel_nsteps = 0
        kernel = nothing
        while done < num_warmup
            nsteps = min(warmup_chunk_size, num_warmup - done)
            if nsteps != kernel_nsteps
                kernel = _compile_warmup_kernel(
                    state,
                    logpdf_fn,
                    nsteps,
                    num_warmup,
                    adapt_step_size,
                    adapt_mass_matrix,
                    mcmc_kwargs,
                )
                kernel_nsteps = nsteps
            end
            state = kernel(state, logpdf_fn, ConcreteRNumber(Int64(done)))
            done += nsteps
            info = _warmup_info(state, done, num_warmup)
            progress_bar && _show_progress(info)
            callback === nothing || callback(info)
        end
    end

    if isnothing(state) && !isnothing(initial_gradient)
        state = MCMCState(
            initial_position,
            initial_gradient,
            initial_potential_energy,
            step_size,
            inverse_mass_matrix,
            rng.seed;
            config=MCMCConfig(; mcmc_kwargs...),
        )
    elseif isnothing(state)
        state = _seed_state(
            rng,
            logpdf_fn,
            initial_position,
            step_size,
            inverse_mass_matrix;
            adapt_step_size=false,
            adapt_mass_matrix=false,
            total_warmup=nothing,
            mcmc_kwargs,
        )
    end

    kernel_nsamp = 0
    kernel = nothing
    while collected < num_samples
        nsamp = min(chunk_size, num_samples - collected)
        if nsamp != kernel_nsamp
            kernel = _compile_sample_kernel(state, logpdf_fn, nsamp, mcmc_kwargs)
            kernel_nsamp = nsamp
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
                isempty(diag_arr) ? nothing : sum(diag_arr) / length(diag_arr)
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

# TODO(#2619): add trace-based mode support
function run_chain(
    rng,
    logpdf_fn::Function,
    initial_position;
    algorithm::Symbol=:NUTS,
    num_warmup::Int=0,
    num_samples::Int=1000,
    chunk_size::Int=100,
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
    mcmc_kwargs = (;
        algorithm,
        max_tree_depth,
        max_delta_energy,
        trajectory_length,
        thinning,
        strong_zero,
    )

    if !progress_bar && isnothing(callback) && num_samples > 0
        monolithic_fn = function (rng, logpdf_fn, pos, grad, pe, ss, imm)
            s, _, _, st = mcmc_logpdf(
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
