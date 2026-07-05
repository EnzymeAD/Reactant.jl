using ..Reactant: TracedRArray

function mh(
    rng::AbstractRNG,
    original_trace,
    original_weight,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    constraint=nothing,
    constrained_addresses=nothing,
    region::Bool=true,
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
                MLIR.IR.Attribute(
                    MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), sym_addr)
                ),
            )
        end
        push!(regenerate_attr, MLIR.IR.Attribute(address_attr))
    end

    trace_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))
    accepted_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))
    log_densities_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    trace_mlir = TracedUtils.get_mlir_data(original_trace)
    weight_mlir = TracedUtils.get_mlir_data(original_weight)

    # Lets a branch flip re-read the observation instead of a stale trace slot.
    constraint_mlir = nothing
    constrained_addresses_attr_val = nothing
    if constraint !== nothing
        @assert constrained_addresses !== nothing "mh: `constrained_addresses` \
            required when `constraint` is provided"
        constrained_addresses_attr = MLIR.IR.Attribute[]
        for address in constrained_addresses
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
            push!(constrained_addresses_attr, MLIR.IR.Attribute(address_attr))
        end
        constraint_mlir = TracedUtils.get_mlir_data(constraint)
        constrained_addresses_attr_val = MLIR.IR.Attribute(constrained_addresses_attr)
    end

    if region
        mh_config_attr = MLIR.IR.Attribute(
            MLIR.API.enzymeMHConfigAttrGet(MLIR.IR.current_context())
        )
        schema_attr = MLIR.IR.Attribute(MLIR.IR.Attribute[MLIR.IR.Attribute("weight")])

        infer_op = MLIR.Dialects.impulse.infer(
            mlir_caller_args,
            MLIR.IR.Value[weight_mlir],
            trace_mlir;
            constraint=constraint_mlir,
            trace=trace_type,
            diagnostics=accepted_type,
            # Zero-filled by the lowering: MH has no per-sample log density (Enzyme#2958).
            log_densities=log_densities_type,
            output_rng_state=mlir_result_types[1],
            state_out=MLIR.IR.Type[weight_type],
            fn=fn_attr,
            selection=MLIR.IR.Attribute(regenerate_attr),
            all_addresses=selection_attr,
            mh_config=mh_config_attr,
            constrained_addresses=constrained_addresses_attr_val,
            state_schema=schema_attr,
            num_samples=Int64(1),
        )

        traced_result = process_probprog_outputs(
            infer_op,
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

        new_trace = TracedRArray{Float64,2}((), MLIR.IR.result(infer_op, 1), (1, pos_size))
        accepted = TracedRArray{Bool,0}((), MLIR.IR.result(infer_op, 2), ())
        new_weight = TracedRArray{Float64,0}((), MLIR.IR.result(infer_op, 5), ())

        return new_trace, new_weight, accepted, traced_result
    end

    mh_op = MLIR.Dialects.impulse.mh(
        trace_mlir,
        weight_mlir,
        mlir_caller_args,
        constraint_mlir;
        new_trace=trace_type,
        new_weight=weight_type,
        accepted=accepted_type,
        output_rng=mlir_result_types[1],
        fn=fn_attr,
        selection=selection_attr,
        regenerate_addresses=MLIR.IR.Attribute(regenerate_attr),
        constrained_addresses=constrained_addresses_attr_val,
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
