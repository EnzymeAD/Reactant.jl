using ..Reactant: ConcreteRNumber, TracedRArray

function hmc(
    rng::AbstractRNG,
    original_trace::Union{ProbProgTrace,TracedRArray{UInt64,0}},
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    mass=nothing,
    step_size=nothing,
    num_steps=nothing,
    initial_momentum=nothing,
) where {Nargs}
    args = (rng, args...)
    mlir_fn_res, argprefix, resprefix, _ = process_probprog_function(f, args, "hmc")

    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    inputs = process_probprog_inputs(linear_args, f, args, fnwrap, argprefix)
    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type

    trace_val = if original_trace isa TracedRArray{UInt64,0}
        MLIR.IR.result(
            MLIR.Dialects.builtin.unrealized_conversion_cast(
                [original_trace.mlir_data]; outputs=[trace_ty]
            ),
            1,
        )
    else
        # First iteration: promote a ProbProgTrace to tensor<ui64>
        promoted = to_trace_tensor(original_trace)
        MLIR.IR.result(
            MLIR.Dialects.builtin.unrealized_conversion_cast(
                [TracedUtils.get_mlir_data(promoted)]; outputs=[trace_ty]
            ),
            1,
        )
    end

    selection_attr = MLIR.IR.Attribute[]
    for address in selection
        address_attr = MLIR.IR.Attribute[]
        for sym in address.path
            sym_addr = reinterpret(UInt64, pointer_from_objref(sym))
            push!(
                address_attr,
                @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
                    MLIR.IR.context()::MLIR.API.MlirContext, sym_addr::UInt64
                )::MLIR.IR.Attribute
            )
        end
        push!(selection_attr, MLIR.IR.Attribute(address_attr))
    end

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type
    accepted_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))

    alg_attr = @ccall MLIR.API.mlir_c.enzymeMCMCAlgorithmAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext,
        0::Int32,  # 0 = HMC
    )::MLIR.IR.Attribute

    mass_val = nothing
    if !isnothing(mass)
        mass_val = TracedUtils.get_mlir_data(mass)
    end

    step_size_val = nothing
    if !isnothing(step_size)
        step_size_val = TracedUtils.get_mlir_data(step_size)
    end

    num_steps_val = nothing
    if !isnothing(num_steps)
        num_steps_val = TracedUtils.get_mlir_data(num_steps)
    end

    initial_momentum_val = nothing
    if !isnothing(initial_momentum)
        initial_momentum_val = TracedUtils.get_mlir_data(initial_momentum)
    end

    hmc_op = MLIR.Dialects.enzyme.mcmc(
        inputs,
        trace_val,
        mass_val;
        step_size=step_size_val,
        num_steps=num_steps_val,
        initial_momentum=initial_momentum_val,
        new_trace=trace_ty,
        accepted=accepted_ty,
        output_rng_state=out_tys[1], # by convention
        alg=alg_attr,
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
    )

    # (new_trace, accepted, output_rng_state)
    process_probprog_outputs(
        hmc_op, linear_results, result, f, args, fnwrap, resprefix, argprefix, 2, true
    )

    new_trace_val = MLIR.IR.result(hmc_op, 1)
    new_trace_ptr = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [new_trace_val]; outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))]
        ),
        1,
    )

    new_trace = TracedRArray{UInt64,0}((), new_trace_ptr, ())
    accepted = TracedRArray{Bool,0}((), MLIR.IR.result(hmc_op, 2), ())

    return new_trace, accepted, result
end
