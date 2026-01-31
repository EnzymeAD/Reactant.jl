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
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args, "mcmc"
    )
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.current_context()::MLIR.API.MlirContext
    )::MLIR.IR.Type

    trace_val = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [TracedUtils.get_mlir_data(original_trace)]; outputs=[trace_ty]
        ),
        1,
    )

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

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.current_context()::MLIR.API.MlirContext
    )::MLIR.IR.Type

    collection_size = num_samples ÷ thinning
    accepted_ty = if collection_size == 1
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
        error("Unknown MCMC algorithm: $algorithm. Supported algorithms are :HMC and :NUTS")
    end

    inverse_mass_matrix_val = nothing
    if !isnothing(inverse_mass_matrix)
        inverse_mass_matrix_val = TracedUtils.get_mlir_data(inverse_mass_matrix)
    end

    step_size_val = nothing
    if !isnothing(step_size)
        step_size_val = TracedUtils.get_mlir_data(step_size)
    end

    mcmc_op = MLIR.Dialects.enzyme.mcmc(
        mlir_caller_args,
        trace_val,
        inverse_mass_matrix_val;
        step_size=step_size_val,
        new_trace=trace_ty,
        accepted=accepted_ty,
        output_rng_state=mlir_result_types[1], # by convention
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
        hmc_config=hmc_config_attr,
        nuts_config=nuts_config_attr,
        num_warmup=Int64(num_warmup),
        num_samples=Int64(num_samples),
        thinning=Int64(thinning),
    )

    # (new_trace, accepted, output_rng_state)
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

    new_trace_val = MLIR.IR.result(mcmc_op, 1)
    new_trace_ptr = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [new_trace_val]; outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))]
        ),
        1,
    )

    new_trace = TracedRArray{UInt64,0}((), new_trace_ptr, ())
    accepted = if collection_size == 1
        TracedRArray{Bool,0}((), MLIR.IR.result(mcmc_op, 2), ())
    else
        TracedRArray{Bool,1}((), MLIR.IR.result(mcmc_op, 2), (collection_size,))
    end

    return new_trace, accepted, traced_result
end
