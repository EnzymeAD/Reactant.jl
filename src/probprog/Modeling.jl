using ..Reactant: MLIR, TracedUtils, AbstractRNG, TracedRArray, to_rarray
using ..Compiler: @compile

include("Utils.jl")

function get_support_kind_int(s::Symbol)
    s === :real && return Int32(0)
    s === :positive && return Int32(1)
    s === :unit_interval && return Int32(2)
    s === :interval && return Int32(3)
    s === :greater_than && return Int32(4)
    s === :less_than && return Int32(5)
    s === :simplex && return Int32(6)
    s === :lower_cholesky && return Int32(7)
    return error("Unknown support type: $s")
end

function sample(
    rng::AbstractRNG,
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    logpdf::Union{Nothing,Function}=nothing,
    support::Symbol=:real,
    bounds::Tuple{Union{Nothing,Real},Union{Nothing,Real}}=(nothing, nothing),
) where {Nargs}
    tt = TRACING_TRACE[]
    is_generative = logpdf === nothing

    if tt !== nothing && is_generative
        push!(tt.address_stack, symbol)
    end

    args_with_rng = (rng, args...)
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args_with_rng, "sample"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)
    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
        MLIR.IR.current_context()::MLIR.API.MlirContext, symbol_addr::UInt64
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

    lower, upper = bounds
    has_lower = !isnothing(lower)
    has_upper = !isnothing(upper)
    lower_val = isnothing(lower) ? NaN : Float64(lower)
    upper_val = isnothing(upper) ? NaN : Float64(upper)

    support_kind = get_support_kind_int(support)
    support_attr = @ccall MLIR.API.mlir_c.enzymeSupportAttrGet(
        MLIR.IR.current_context()::MLIR.API.MlirContext,
        support_kind::Int32,
        lower_val::Float64,
        has_lower::Bool,
        upper_val::Float64,
        has_upper::Bool,
    )::MLIR.IR.Attribute

    sample_op = MLIR.Dialects.enzyme.sample(
        mlir_caller_args;
        outputs=mlir_result_types,
        fn=fn_attr,
        logpdf=logpdf_attr,
        symbol=symbol_attr,
        support=support_attr,
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

    if tt !== nothing
        if !is_generative
            sample_shape = size(mlir_result_types[2])
            num_el = max(1, prod(sample_shape))
            entry = TraceEntry(
                symbol, sample_shape, num_el, tt.position_size, copy(tt.address_stack)
            )
            push!(tt.entries, entry)
            tt.position_size += num_el
        else
            pop!(tt.address_stack)
        end
    end

    return traced_result
end

function sample(
    rng::AbstractRNG, dist::D; symbol::Symbol=gensym("sample")
) where {D<:Distribution}
    dist_type = typeof(dist)
    return sample(
        rng,
        sampler(dist_type),
        params(dist)...;
        symbol=symbol,
        logpdf=logpdf_fn(dist_type),
        support=support(dist_type),
        bounds=bounds(dist_type),
    )
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

function simulate(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    args = (rng, args...)

    existing_tt = TRACING_TRACE[]
    tt = existing_tt !== nothing ? existing_tt : TracedTrace()

    ppf = if existing_tt !== nothing
        process_probprog_function(f, args, "simulate")
    else
        scoped_with(TRACING_TRACE => tt) do
            process_probprog_function(f, args, "simulate")
        end
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

    trace_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    simulate_op = MLIR.Dialects.enzyme.simulate(
        mlir_caller_args;
        trace=trace_type,
        weight=weight_type,
        outputs=mlir_result_types,
        fn=fn_attr,
        selection=selection_attr,
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

    trace_val = TracedRArray{Float64,2}((), MLIR.IR.result(simulate_op, 1), (1, pos_size))
    weight_val = TracedRArray{Float64,0}((), MLIR.IR.result(simulate_op, 2), ())

    return trace_val, weight_val, traced_result
end

# Gen-like helper function.
function simulate_(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    tt = TracedTrace()

    compiled_fn = scoped_with(TRACING_TRACE => tt) do
        @compile optimize = :probprog simulate(rng, f, args...)
    end
    trace_tensor, weight_val, traced_result = compiled_fn(rng, f, args...)

    retval = traced_result[2:end]

    trace = unflatten_trace(trace_tensor, weight_val, tt.entries, retval)
    return trace, trace.weight
end

# Gen-like helper function.
function generate_(
    rng::AbstractRNG, constraint::Constraint, f::Function, args::Vararg{Any,Nargs}
) where {Nargs}
    tt = TracedTrace()
    constrained_addresses = extract_addresses(constraint)

    constraint_flat = Float64[]
    for addr in constrained_addresses
        append!(constraint_flat, vec(constraint[addr]))
    end
    constraint_tensor = to_rarray(reshape(constraint_flat, 1, :))

    compiled_fn = scoped_with(TRACING_TRACE => tt) do
        @compile optimize = :probprog generate(
            rng, constraint_tensor, f, args...; constrained_addresses
        )
    end
    trace_tensor, weight_val, traced_result = compiled_fn(
        rng, constraint_tensor, f, args...
    )
    retval = traced_result[2:end]

    trace = unflatten_trace(trace_tensor, weight_val, tt.entries, retval)
    return trace, trace.weight
end

function generate(
    rng::AbstractRNG,
    constraint_tensor,
    f::Function,
    args::Vararg{Any,Nargs};
    constrained_addresses::Set{Address},
) where {Nargs}
    args = (rng, args...)

    existing_tt = TRACING_TRACE[]
    tt = existing_tt !== nothing ? existing_tt : TracedTrace()

    ppf = if existing_tt !== nothing
        process_probprog_function(f, args, "generate")
    else
        scoped_with(TRACING_TRACE => tt) do
            process_probprog_function(f, args, "generate")
        end
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

    constrained_addresses_attr = MLIR.IR.Attribute[]
    for address in constrained_addresses
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
        push!(constrained_addresses_attr, MLIR.IR.Attribute(address_attr))
    end

    selection_attr = build_selection_attr(tt)
    pos_size = tt.position_size

    trace_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    constraint_mlir = TracedUtils.get_mlir_data(constraint_tensor)

    generate_op = MLIR.Dialects.enzyme.generate(
        mlir_caller_args,
        constraint_mlir;
        trace=trace_type,
        weight=weight_type,
        outputs=mlir_result_types,
        fn=fn_attr,
        selection=selection_attr,
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

    trace_val = TracedRArray{Float64,2}((), MLIR.IR.result(generate_op, 1), (1, pos_size))
    weight_val = TracedRArray{Float64,0}((), MLIR.IR.result(generate_op, 2), ())

    return trace_val, weight_val, traced_result
end
