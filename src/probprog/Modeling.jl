using ..Reactant: MLIR, TracedUtils, AbstractRNG, TracedRArray, ConcreteRNumber
using ..Compiler: @jit, @compile

include("Utils.jl")

function sample(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    logpdf::Union{Nothing,Function}=nothing,
) where {Nargs}
    args_with_rng = (rng, args...)
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args_with_rng, "sample"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)
    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext, symbol_addr::UInt64
    )::MLIR.IR.Attribute

    logpdf_attr = nothing
    if logpdf isa Function
        samples = f(args_with_rng...)

        # Logpdf calling convention: `(sample, args...)` (no rng state)
        logpdf_args = (samples, args...)

        logpdf_attr = MLIR.IR.FlatSymbolRefAttribute(
            process_probprog_function(logpdf, logpdf_args, "logpdf", false).f_name
        )
    end

    sample_op = MLIR.Dialects.enzyme.sample(
        mlir_caller_args;
        outputs=mlir_result_types,
        fn=fn_attr,
        logpdf=logpdf_attr,
        symbol=symbol_attr,
        name=Base.String(symbol),
    )

    traced_result = process_probprog_outputs(
        sample_op,
        linear_results,
        traced_result,
        f,
        args_with_rng,
        fnwrapped,
        resprefix,
        argprefix,
    )

    return traced_result
end

function untraced_call(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    args_with_rng = (rng, args...)

    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args_with_rng, "untraced_call"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)

    call_op = MLIR.Dialects.enzyme.untracedCall(
        mlir_caller_args; outputs=mlir_result_types, fn=fn_attr
    )

    traced_result = process_probprog_outputs(
        call_op,
        linear_results,
        traced_result,
        f,
        args_with_rng,
        fnwrapped,
        resprefix,
        argprefix,
    )

    return traced_result
end

# Gen-like helper function.
function simulate_(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    trace = nothing

    compiled_fn = @compile optimize = :probprog simulate(rng, f, args...)

    seed_buffer = only(rng.seed.data).buffer
    GC.@preserve seed_buffer begin
        t, _, _ = compiled_fn(rng, f, args...)
        trace = from_trace_tensor(t)
    end

    return trace, trace.weight
end

function simulate(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    args = (rng, args...)
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args, "simulate"
    )
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type
    weight_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    simulate_op = MLIR.Dialects.enzyme.simulate(
        mlir_caller_args;
        trace=trace_ty,
        weight=weight_ty,
        outputs=mlir_result_types,
        fn=fn_attr,
    )

    traced_result = process_probprog_outputs(
        simulate_op,
        linear_results,
        traced_result,
        f,
        args,
        fnwrapped,
        resprefix,
        argprefix,
        2,
    )

    trace = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(simulate_op, 1)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))],
        ),
        1,
    )

    trace = TracedRArray{UInt64,0}((), trace, ())
    weight = TracedRArray{Float64,0}((), MLIR.IR.result(simulate_op, 2), ())

    return trace, weight, traced_result
end

# Gen-like helper function.
function generate_(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    constraint::Constraint=Constraint(),
) where {Nargs}
    trace = nothing

    constraint_ptr = ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(constraint)))

    constrained_addresses = extract_addresses(constraint)

    function wrapper_fn(rng, constraint_ptr, args...)
        return generate(rng, f, args...; constraint_ptr, constrained_addresses)
    end

    compiled_fn = @compile optimize = :probprog wrapper_fn(rng, constraint_ptr, args...)

    seed_buffer = only(rng.seed.data).buffer
    GC.@preserve seed_buffer constraint begin
        t, _, _ = compiled_fn(rng, constraint_ptr, args...)
        trace = from_trace_tensor(t)
    end

    return trace, trace.weight
end

function generate(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    constraint_ptr::TracedRNumber,
    constrained_addresses::Set{Address},
) where {Nargs}
    args = (rng, args...)

    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args, "generate"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)

    constraint_ty = @ccall MLIR.API.mlir_c.enzymeConstraintTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type

    constraint_val = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [TracedUtils.get_mlir_data(constraint_ptr)]; outputs=[constraint_ty]
        ),
        1,
    )

    constrained_addresses_attr = MLIR.IR.Attribute[]
    for address in constrained_addresses
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
        push!(constrained_addresses_attr, MLIR.IR.Attribute(address_attr))
    end

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type
    weight_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    generate_op = MLIR.Dialects.enzyme.generate(
        mlir_caller_args,
        constraint_val;
        trace=trace_ty,
        weight=weight_ty,
        outputs=mlir_result_types,
        fn=fn_attr,
        constrained_addresses=MLIR.IR.Attribute(constrained_addresses_attr),
    )

    traced_result = process_probprog_outputs(
        generate_op,
        linear_results,
        traced_result,
        f,
        args,
        fnwrapped,
        resprefix,
        argprefix,
        2,
    )

    trace = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(generate_op, 1)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))],
        ),
        1,
    )

    trace = TracedRArray{UInt64,0}((), trace, ())
    weight = TracedRArray{Float64,0}((), MLIR.IR.result(generate_op, 2), ())

    return trace, weight, traced_result
end
