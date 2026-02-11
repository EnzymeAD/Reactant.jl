using ..Reactant: TracedRArray

function mh(
    rng::AbstractRNG,
    original_trace,
    original_weight,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
) where {Nargs}
    args = (rng, args...)

    tt = TracedTrace()
    ppf = ScopedValues.with(TRACING_TRACE => tt) do
        process_probprog_function(f, args, "mh")
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
    selection_attr = build_selection_attr(tt)
    pos_size = tt.position_size

    regenerate_attr = MLIR.IR.Attribute[]
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
        push!(regenerate_attr, MLIR.IR.Attribute(address_attr))
    end

    trace_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))
    accepted_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))

    trace_mlir = TracedUtils.get_mlir_data(original_trace)
    weight_mlir = TracedUtils.get_mlir_data(original_weight)

    mh_op = MLIR.Dialects.enzyme.mh(
        trace_mlir,
        weight_mlir,
        mlir_caller_args;
        new_trace=trace_type,
        new_weight=weight_type,
        accepted=accepted_type,
        output_rng=mlir_result_types[1],
        fn=fn_attr,
        selection=selection_attr,
        regenerate_addresses=MLIR.IR.Attribute(regenerate_attr),
    )

    traced_result = process_probprog_outputs(
        mh_op,
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

    new_trace = TracedRArray{Float64,2}((), MLIR.IR.result(mh_op, 1), (1, pos_size))
    new_weight = TracedRArray{Float64,0}((), MLIR.IR.result(mh_op, 2), ())
    accepted = TracedRArray{Bool,0}((), MLIR.IR.result(mh_op, 3), ())

    return new_trace, new_weight, accepted, traced_result
end

const metropolis_hastings = mh
