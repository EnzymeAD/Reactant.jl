using ..Reactant: TracedRArray

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
    ppf = scoped_with(TRACING_TRACE => tt) do
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
                @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
                    MLIR.IR.current_context()::MLIR.API.MlirContext, sym_addr::UInt64
                )::MLIR.IR.Attribute
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

    hmc_config_attr = nothing
    nuts_config_attr = nothing

    if algorithm == :HMC
        hmc_config_attr = @ccall MLIR.API.mlir_c.enzymeHMCConfigAttrGet(
            MLIR.IR.current_context()::MLIR.API.MlirContext,
            trajectory_length::Float64,
            adapt_step_size::Bool,
            adapt_mass_matrix::Bool,
        )::MLIR.IR.Attribute
    elseif algorithm == :NUTS
        nuts_config_attr = @ccall MLIR.API.mlir_c.enzymeNUTSConfigAttrGet(
            MLIR.IR.current_context()::MLIR.API.MlirContext,
            max_tree_depth::Int64,
            max_delta_energy::Float64,
            adapt_step_size::Bool,
            adapt_mass_matrix::Bool,
        )::MLIR.IR.Attribute
    else
        error("Unknown MCMC algorithm: $algorithm. Supported: :HMC, :NUTS")
    end

    inverse_mass_matrix_val = if isnothing(inverse_mass_matrix)
        nothing
    else
        TracedUtils.get_mlir_data(inverse_mass_matrix)
    end
    step_size_val = isnothing(step_size) ? nothing : TracedUtils.get_mlir_data(step_size)

    trace_val = TracedUtils.get_mlir_data(original_trace)

    mcmc_op = MLIR.Dialects.enzyme.mcmc(
        mlir_caller_args,
        trace_val,
        inverse_mass_matrix_val;
        step_size=step_size_val,
        trace=trace_type,
        diagnostics=diagnostics_type,
        output_rng_state=mlir_result_types[1],
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

    return new_trace, diagnostics, traced_result
end
