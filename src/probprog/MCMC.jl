using ..Reactant: TracedRArray, ReactantRNG
using Serialization

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
        process_probprog_function(f, args, "mcmc")
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
    diagnostics_type = if collection_size == 1
        MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))
    else
        MLIR.IR.TensorType([Int64(collection_size)], MLIR.IR.Type(Bool))
    end

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

    mcmc_op = MLIR.Dialects.enzyme.mcmc(
        mlir_caller_args,
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
    diagnostics = if collection_size == 1
        TracedRArray{Bool,0}((), MLIR.IR.result(mcmc_op, 2), ())
    else
        TracedRArray{Bool,1}((), MLIR.IR.result(mcmc_op, 2), (collection_size,))
    end

    inv_mass_shape =
        isnothing(inverse_mass_matrix) ? (1, selection_pos_size) : size(inverse_mass_matrix)

    state = MCMCState(
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 4), (1, selection_pos_size)),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 5), (1, selection_pos_size)),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 6), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 7), ()),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 8), inv_mass_shape),
        TracedRArray{UInt64,1}((), MLIR.IR.result(mcmc_op, 3), size(rng.seed)),
    )

    return new_trace, diagnostics, traced_result, state
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
) where {Nargs}
    pos_size = length(initial_position)

    sample_pos = TracedRArray{Float64,2}((), nothing, (1, pos_size))
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

    inverse_mass_matrix_val = if isnothing(inverse_mass_matrix)
        nothing
    else
        TracedUtils.get_mlir_data(inverse_mass_matrix)
    end
    step_size_val = isnothing(step_size) ? nothing : TracedUtils.get_mlir_data(step_size)
    initial_position_val = TracedUtils.get_mlir_data(initial_position)
    initial_gradient_val =
        isnothing(initial_gradient) ? nothing : TracedUtils.get_mlir_data(initial_gradient)
    initial_pe_val = if isnothing(initial_potential_energy)
        nothing
    else
        TracedUtils.get_mlir_data(initial_potential_energy)
    end

    collection_size = num_samples ÷ thinning
    trace_type = MLIR.IR.TensorType([collection_size, pos_size], MLIR.IR.Type(Float64))
    diagnostics_type = if collection_size == 1
        MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))
    else
        MLIR.IR.TensorType([Int64(collection_size)], MLIR.IR.Type(Bool))
    end

    position_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    scalar_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    inv_mass_type = if isnothing(inverse_mass_matrix_val)
        position_type
    else
        MLIR.IR.type(inverse_mass_matrix_val)
    end

    mcmc_op = MLIR.Dialects.enzyme.mcmc(
        mlir_caller_args;
        inverse_mass_matrix=inverse_mass_matrix_val,
        step_size=step_size_val,
        initial_position=initial_position_val,
        initial_gradient=initial_gradient_val,
        initial_potential_energy=initial_pe_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        output_rng_state=mlir_result_types[1],
        final_position=position_type,
        final_gradient=position_type,
        final_potential_energy=scalar_type,
        final_step_size=scalar_type,
        final_inverse_mass_matrix=inv_mass_type,
        logpdf_fn=logpdf_fn_attr,
        selection=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
        all_addresses=MLIR.IR.Attribute(MLIR.IR.Attribute[]),
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
    diagnostics = if collection_size == 1
        TracedRArray{Bool,0}((), MLIR.IR.result(mcmc_op, 2), ())
    else
        TracedRArray{Bool,1}((), MLIR.IR.result(mcmc_op, 2), (collection_size,))
    end

    inv_mass_shape =
        isnothing(inverse_mass_matrix) ? (1, pos_size) : size(inverse_mass_matrix)

    state = MCMCState(
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 4), (1, pos_size)),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 5), (1, pos_size)),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 6), ()),
        TracedRArray{Float64,0}((), MLIR.IR.result(mcmc_op, 7), ()),
        TracedRArray{Float64,2}((), MLIR.IR.result(mcmc_op, 8), inv_mass_shape),
        TracedRArray{UInt64,1}((), MLIR.IR.result(mcmc_op, 3), size(rng.seed)),
    )

    return new_trace, diagnostics, traced_result, state
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
    num_warmup::Int=0,
    adapt_step_size::Bool=false,
    adapt_mass_matrix::Bool=false,
    kwargs...,
) where {Nargs}
    return mcmc_logpdf(
        ReactantRNG(state.rng),
        logdensity_fn,
        state.position,
        args...;
        inverse_mass_matrix,
        step_size,
        initial_gradient,
        initial_potential_energy,
        num_warmup,
        adapt_step_size,
        adapt_mass_matrix,
        kwargs...,
    )
end

function _print_progress(current, total; width=40)
    frac = current / total
    filled = round(Int, frac * width)
    bar = "━"^filled * " "^(width - filled)
    print("\r  Sampling $(bar) $(current)/$(total)")
    current == total && println()
    flush(stdout)
end

function save_state(filename::String, state::MCMCState)
    data = Dict{String,Any}(
        "position" => Array(state.position),
        "gradient" => Array(state.gradient),
        "potential_energy" => Array(state.potential_energy)[],
        "step_size" => Array(state.step_size)[],
        "inverse_mass_matrix" => Array(state.inverse_mass_matrix),
        "rng" => Array(state.rng),
    )
    open(io -> Serialization.serialize(io, data), filename, "w")
end

function load_state(filename::String)
    data = open(Serialization.deserialize, filename)
    return MCMCState(
        Reactant.to_rarray(data["position"]),
        Reactant.to_rarray(data["gradient"]),
        Reactant.to_rarray(fill(data["potential_energy"])),
        Reactant.to_rarray(fill(data["step_size"])),
        Reactant.to_rarray(data["inverse_mass_matrix"]),
        Reactant.to_rarray(data["rng"]),
    )
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
    progress_bar::Bool=true,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    thinning::Int=1,
    trajectory_length::Float64=2π,
)
    first_chunk = min(chunk_size, num_samples)
    remaining = num_samples - first_chunk

    warmup_fn = function (rng, logpdf_fn, pos, ss, imm)
        samples, _, _, state = mcmc_logpdf(
            rng, logpdf_fn, pos;
            algorithm, step_size=ss, inverse_mass_matrix=imm,
            max_tree_depth, max_delta_energy, trajectory_length,
            num_warmup, num_samples=first_chunk, thinning,
            adapt_step_size, adapt_mass_matrix,
        )
        return samples, state
    end

    progress_bar && num_warmup > 0 && print("  Warmup: $(num_warmup) steps... ")

    compiled_warmup = Reactant.Compiler.compile(
        warmup_fn, (rng, logpdf_fn, initial_position, step_size, inverse_mass_matrix);
        optimize=:probprog,
    )
    samples1, state = compiled_warmup(
        rng, logpdf_fn, initial_position, step_size, inverse_mass_matrix,
    )

    progress_bar && num_warmup > 0 && println("done.")

    all_chunks = [Array(samples1)]
    progress_bar && _print_progress(first_chunk, num_samples)

    if remaining > 0
        full_chunks = remaining ÷ chunk_size
        remainder = remaining % chunk_size

        if full_chunks > 0
            continue_fn = function (sr, lf, pos, grad, pe, ss, imm)
                samples, _, _, st = mcmc_logpdf(
                    ReactantRNG(sr), lf, pos;
                    algorithm, step_size=ss, inverse_mass_matrix=imm,
                    initial_gradient=grad, initial_potential_energy=pe,
                    max_tree_depth, max_delta_energy, trajectory_length,
                    num_warmup=0, num_samples=chunk_size, thinning,
                    adapt_step_size=false, adapt_mass_matrix=false,
                )
                return samples, st
            end

            compiled_continue = Reactant.Compiler.compile(
                continue_fn,
                (state.rng, logpdf_fn, state.position, state.gradient,
                 state.potential_energy, state.step_size, state.inverse_mass_matrix);
                optimize=:probprog,
            )

            for i in 1:full_chunks
                chunk, state = compiled_continue(
                    state.rng, logpdf_fn, state.position, state.gradient,
                    state.potential_energy, state.step_size, state.inverse_mass_matrix,
                )
                push!(all_chunks, Array(chunk))
                progress_bar && _print_progress(first_chunk + i * chunk_size, num_samples)
            end
        end

        if remainder > 0
            remainder_fn = function (sr, lf, pos, grad, pe, ss, imm)
                samples, _, _, st = mcmc_logpdf(
                    ReactantRNG(sr), lf, pos;
                    algorithm, step_size=ss, inverse_mass_matrix=imm,
                    initial_gradient=grad, initial_potential_energy=pe,
                    max_tree_depth, max_delta_energy, trajectory_length,
                    num_warmup=0, num_samples=remainder, thinning,
                    adapt_step_size=false, adapt_mass_matrix=false,
                )
                return samples, st
            end

            compiled_remainder = Reactant.Compiler.compile(
                remainder_fn,
                (state.rng, logpdf_fn, state.position, state.gradient,
                 state.potential_energy, state.step_size, state.inverse_mass_matrix);
                optimize=:probprog,
            )

            chunk, state = compiled_remainder(
                state.rng, logpdf_fn, state.position, state.gradient,
                state.potential_energy, state.step_size, state.inverse_mass_matrix,
            )
            push!(all_chunks, Array(chunk))
            progress_bar && _print_progress(num_samples, num_samples)
        end
    end

    return vcat(all_chunks...), state
end

function run_chain(
    state::MCMCState,
    logpdf_fn::Function;
    algorithm::Symbol=:NUTS,
    num_samples::Int=1000,
    chunk_size::Int=100,
    progress_bar::Bool=true,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    thinning::Int=1,
    trajectory_length::Float64=2π,
)
    return run_chain(
        ReactantRNG(state.rng), logpdf_fn, state.position;
        algorithm, num_warmup=0, num_samples, chunk_size,
        step_size=state.step_size,
        inverse_mass_matrix=state.inverse_mass_matrix,
        progress_bar, max_tree_depth, max_delta_energy,
        adapt_step_size=false, adapt_mass_matrix=false,
        thinning, trajectory_length,
    )
end
