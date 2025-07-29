using ..Reactant:
    MLIR, TracedUtils, AbstractRNG, AbstractConcreteArray, TracedRArray, ConcreteRNumber
using ..Compiler: @jit, @compile

function process_mlir_function(f::Function, args::Tuple, op_name::String)
    argprefix = gensym(op_name * "arg")
    resprefix = gensym(op_name * "result")
    resargprefix = gensym(op_name * "resarg")

    wrapper_fn = (all_args...) -> begin
        res = f(all_args...)
        (all_args[1], (res isa Tuple ? res : (res,))...)
    end

    mlir_fn_res = invokelatest(
        TracedUtils.make_mlir_fn,
        wrapper_fn,
        args,
        (),
        string(f),
        false;
        do_transpose=false,
        args_in_result=:result,
        argprefix,
        resprefix,
        resargprefix,
    )

    return mlir_fn_res, argprefix, resprefix, resargprefix
end

function process_mlir_inputs(linear_args, f, args, fnwrap, argprefix)
    inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 2 && fnwrap
            TracedUtils.push_val!(inputs, f, path[3:end])
        else
            if fnwrap && idx > 1
                idx -= 1
            end
            TracedUtils.push_val!(inputs, args[idx], path[3:end])
        end
    end
    return inputs
end

function process_mlir_outputs(
    op, linear_results, result, f, args, fnwrap, resprefix, argprefix, start_idx=0
)
    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(op, i + start_idx)

        if TracedUtils.has_idx(res, resprefix)
            path = TracedUtils.get_idx(res, resprefix)
            TracedUtils.set!(result, path[2:end], resv)
        end

        if TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if fnwrap && idx == 2
                TracedUtils.set!(f, path[3:end], resv)
            else
                if fnwrap && idx > 2
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], resv)
            end
        end

        if !TracedUtils.has_idx(res, resprefix) && !TracedUtils.has_idx(res, argprefix)
            TracedUtils.set!(res, (), resv)
        end
    end
end

function sample(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    logpdf::Union{Nothing,Function}=nothing,
) where {Nargs}
    res = sample_internal(rng, f, args...; symbol, logpdf)

    @assert res isa Tuple && length(res) >= 1 && res[1] isa AbstractRNG "Expected first result to be RNG"

    res = res[2:end]

    return length(res) == 1 ? res[1] : res
end

function sample_internal(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    logpdf::Union{Nothing,Function}=nothing,
) where {Nargs}
    args = (rng, args...)
    mlir_fn_res, argprefix, resprefix, resargprefix = process_mlir_function(
        f, args, "sample"
    )

    (; result, linear_args, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    inputs = process_mlir_inputs(linear_args, f, args, fnwrap, argprefix)
    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    sym = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(sym))

    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext, symbol_addr::UInt64
    )::MLIR.IR.Attribute

    # Construct MLIR attribute if Julia logpdf function is provided.
    logpdf_attr = nothing
    if logpdf !== nothing
        # Just to get static information about the sample. TODO: kwargs?
        example_sample = f(args...)

        # Remove AbstractRNG from `f`'s argument list if present, assuming that
        # logpdf parameters follows `(sample, args...)` convention.
        logpdf_args = nothing
        if !isempty(args) && args[1] isa AbstractRNG
            logpdf_args = (example_sample, Base.tail(args)...)  # TODO: kwargs?
        else
            logpdf_args = (example_sample, args...)
        end

        logpdf_mlir = invokelatest(
            TracedUtils.make_mlir_fn,
            logpdf,
            logpdf_args,
            (),
            string(logpdf),
            false;
            do_transpose=false,
            args_in_result=:result,
        )

        logpdf_sym = TracedUtils.get_attribute_by_name(logpdf_mlir.f, "sym_name")
        logpdf_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(logpdf_sym))
    end

    sample_op = MLIR.Dialects.enzyme.sample(
        inputs;
        outputs=out_tys,
        fn=fn_attr,
        logpdf=logpdf_attr,
        symbol=symbol_attr,
        name=Base.String(symbol),
    )

    process_mlir_outputs(
        sample_op, linear_results, result, f, args, fnwrap, resprefix, argprefix
    )

    return result
end

function call(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    res = @jit optimize = :probprog call_internal(rng, f, args...)

    res = map(res[2:end]) do r
        r isa AbstractConcreteArray ? Array(r) : r
    end

    return length(res) == 1 ? res[1] : res
end

function call_internal(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    args = (rng, args...)
    mlir_fn_res, argprefix, resprefix, resargprefix = process_mlir_function(f, args, "call")

    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    inputs = process_mlir_inputs(linear_args, f, args, fnwrap, argprefix)
    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    call_op = MLIR.Dialects.enzyme.untracedCall(inputs; outputs=out_tys, fn=fn_attr)

    process_mlir_outputs(
        call_op, linear_results, result, f, args, fnwrap, resprefix, argprefix
    )

    return result
end

function simulate(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    trace = nothing

    compiled_fn = @compile optimize = :probprog simulate_internal(rng, f, args...)

    old_gc_state = GC.enable(false)
    try
        trace, _, _ = compiled_fn(rng, f, args...)
    finally
        GC.enable(old_gc_state)
    end

    trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace)[1]))

    trace.rng = rng
    trace.fn = f
    trace.args = args

    return trace, trace.weight
end

function simulate_internal(
    rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}
) where {Nargs}
    args = (rng, args...)
    mlir_fn_res, argprefix, resprefix, resargprefix = process_mlir_function(
        f, args, "simulate"
    )

    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    inputs = process_mlir_inputs(linear_args, f, args, fnwrap, argprefix)
    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type
    weight_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    simulate_op = MLIR.Dialects.enzyme.simulate(
        inputs; trace=trace_ty, weight=weight_ty, outputs=out_tys, fn=fn_attr
    )

    process_mlir_outputs(
        simulate_op, linear_results, result, f, args, fnwrap, resprefix, argprefix, 2
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

    return trace, weight, result
end

function generate(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    constraint::Constraint=Constraint(),
) where {Nargs}
    trace = nothing

    constraint_ptr = ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(constraint)))

    constrained_addresses = extract_addresses(constraint)

    function wrapper_fn(rng, constraint_ptr, args...)
        return generate_internal(rng, f, args...; constraint_ptr, constrained_addresses)
    end

    compiled_fn = @compile optimize = :probprog wrapper_fn(rng, constraint_ptr, args...)

    old_gc_state = GC.enable(false)
    try
        trace, _, _ = compiled_fn(rng, constraint_ptr, args...)
    finally
        GC.enable(old_gc_state)
    end

    trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace)[1]))

    trace.rng = rng
    trace.fn = f
    trace.args = args

    return trace, trace.weight
end

function generate_internal(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    constraint_ptr::TracedRNumber,
    constrained_addresses::Set{Address},
) where {Nargs}
    args = (rng, args...)
    mlir_fn_res, argprefix, resprefix, resargprefix = process_mlir_function(
        f, args, "generate"
    )

    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    inputs = process_mlir_inputs(linear_args, f, args, fnwrap, argprefix)
    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

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
        inputs,
        constraint_val;
        trace=trace_ty,
        weight=weight_ty,
        outputs=out_tys,
        fn=fn_attr,
        constrained_addresses=MLIR.IR.Attribute(constrained_addresses_attr),
    )

    process_mlir_outputs(
        generate_op, linear_results, result, f, args, fnwrap, resprefix, argprefix, 2
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

    return trace, weight, result
end
