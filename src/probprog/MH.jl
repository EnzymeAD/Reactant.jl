using ..Reactant: TracedRArray

function mh(
    rng::AbstractRNG,
    original_trace,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
) where {Nargs}
    args = (rng, args...)
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args, "mh"
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
    accepted_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))

    mh_op = MLIR.Dialects.enzyme.mh(
        mlir_caller_args,
        trace_val;
        new_trace=trace_ty,
        accepted=accepted_ty,
        output_rng_state=mlir_result_types[1], # by convention
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
    )

    # Return (new_trace, accepted, output_rng_state)
    traced_result = process_probprog_outputs(
        mh_op,
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

    new_trace_val = MLIR.IR.result(mh_op, 1)
    new_trace_ptr = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [new_trace_val]; outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))]
        ),
        1,
    )

    new_trace = TracedRArray{UInt64,0}((), new_trace_ptr, ())
    accepted = TracedRArray{Bool,0}((), MLIR.IR.result(mh_op, 2), ())

    return new_trace, accepted, traced_result
end

const metropolis_hastings = mh
