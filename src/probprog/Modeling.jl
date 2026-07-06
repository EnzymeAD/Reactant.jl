using ..Reactant: MLIR, TracedUtils, AbstractRNG, TracedRArray, to_rarray
using ..Compiler: @compile

include("Utils.jl")

function get_support_kind(s::Symbol)
    s === :real && return MLIR.API.EnzymeSupportKind_Real
    s === :positive && return MLIR.API.EnzymeSupportKind_Positive
    s === :unit_interval && return MLIR.API.EnzymeSupportKind_UnitInterval
    s === :interval && return MLIR.API.EnzymeSupportKind_Interval
    s === :greater_than && return MLIR.API.EnzymeSupportKind_GreaterThan
    s === :less_than && return MLIR.API.EnzymeSupportKind_LessThan
    s === :simplex && return MLIR.API.EnzymeSupportKind_Simplex
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
    dist::Union{Nothing,AbstractString}=nothing,
) where {Nargs}
    tt = TRACING_TRACE[]
    is_generative = logpdf === nothing && dist === nothing

    if tt !== nothing && is_generative
        push!(tt.address_stack, symbol)
    end

    args_with_rng = (rng, args...)
    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args_with_rng, "sample"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)
    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), symbol_addr)
    )

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

    sample_shape_full = size(mlir_result_types[2])
    event_size = support === :simplex ? Int64(last(sample_shape_full)) : Int64(0)

    support_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeSupportAttrGet(
            MLIR.IR.current_context(),
            get_support_kind(support),
            has_lower,
            lower_val,
            has_upper,
            upper_val,
            event_size,
        ),
    )

    dist_attr =
        dist === nothing ? nothing : Base.parse(MLIR.IR.Attribute, Base.String(dist))

    sample_op = MLIR.Dialects.impulse.sample(
        mlir_caller_args;
        outputs=mlir_result_types,
        fn=fn_attr,
        logpdf=logpdf_attr,
        symbol=symbol_attr,
        support=support_attr,
        dist=dist_attr,
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
                symbol,
                sample_shape,
                num_el,
                tt.position_size,
                copy(tt.address_stack),
                support,
                Int(event_size),
            )
            push!(tt.entries, entry)
            tt.position_size += num_el
        else
            pop!(tt.address_stack)
        end
    end

    return traced_result
end

_sample_operand(x::MLIR.IR.Value) = x
_sample_operand(x::Reactant.TracedRNumber) = TracedUtils.get_mlir_data(x)
_sample_operand(x::Reactant.TracedRArray) = TracedUtils.get_mlir_data(x)
_sample_operand(x::Real) = TracedUtils.get_mlir_data(Ops.constant(Float64(x)))
function _sample_operand(x::AbstractArray{<:Real})
    return TracedUtils.get_mlir_data(Ops.constant(Array{Float64}(x)))
end

_dist_result(value, shape::Tuple{}) = Reactant.TracedRNumber{Float64}((), value)
_dist_result(value, shape::Tuple) = TracedRArray{Float64,length(shape)}((), value, shape)

function _sample_from_dist(rng::AbstractRNG, dist::AbstractDistribution, symbol::Symbol)
    dist_type = typeof(dist)
    sup = support(dist_type)
    shp = dist_shape(dist)

    rng_mlir = TracedUtils.get_mlir_data(rng.seed)
    inputs = MLIR.IR.Value[rng_mlir]
    for p in dist_operands(dist)
        push!(inputs, _sample_operand(p))
    end

    value_type = MLIR.IR.TensorType(collect(Int, shp), MLIR.IR.Type(Float64))
    result_types = MLIR.IR.Type[MLIR.IR.type(rng_mlir), value_type]

    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), symbol_addr)
    )
    support_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeSupportAttrGet(
            MLIR.IR.current_context(),
            get_support_kind(sup),
            false,
            NaN,
            false,
            NaN,
            Int64(0),
        ),
    )
    dist_attr = Base.parse(MLIR.IR.Attribute, dist_spec(dist_type))

    sample_op = MLIR.Dialects.impulse.sample(
        inputs;
        outputs=result_types,
        fn=nothing,
        logpdf=nothing,
        symbol=symbol_attr,
        support=support_attr,
        dist=dist_attr,
        name=Base.String(symbol),
    )

    TracedUtils.set_mlir_data!(rng.seed, MLIR.IR.result(sample_op, 1))
    value = _dist_result(MLIR.IR.result(sample_op, 2), shp)

    tt = TRACING_TRACE[]
    if tt !== nothing
        num_el = max(1, prod(shp))
        entry = TraceEntry(
            symbol, shp, num_el, tt.position_size, copy(tt.address_stack), sup, 0
        )
        push!(tt.entries, entry)
        tt.position_size += num_el
    end

    return rng, value
end

# Constants would be baked into the traced sampler, leaving the emitted logpdf no operands.
_traced_param(x::Reactant.TracedRNumber) = x
_traced_param(x::Reactant.TracedRArray) = x
_traced_param(x::Reactant.ConcreteRNumber) = x
_traced_param(x::Reactant.ConcreteRArray) = x
_traced_param(x::Real) = Ops.constant(Float64(x))
_traced_param(x::AbstractArray{<:Real}) = Ops.constant(Array{Float64}(x))

function sample(
    rng::AbstractRNG, dist::D; symbol::Symbol=gensym("sample")
) where {D<:AbstractDistribution}
    if sampler_traced(dist)
        return sample(
            rng,
            sampler(D),
            map(_traced_param, dist_operands(dist))...,
            dist_shape(dist);
            symbol=symbol,
            logpdf=nothing,
            support=support(D),
            bounds=bounds(D),
            dist=dist_spec(D),
        )
    end
    return _sample_from_dist(rng, dist, symbol)
end

function _if_wrap(template::TracedRArray{T,N}, value) where {T,N}
    return TracedRArray{T,N}((), value, size(template))
end
function _if_wrap(template::Reactant.TracedRNumber{T}, value) where {T}
    return Reactant.TracedRNumber{T}((), value)
end

# Emits impulse.if (not stablehlo.if) so expand-impulse threads the trace through it.
# Each thunk takes no arguments; both must return the same MLIR type.
function if_(rng::AbstractRNG, cond, then_thunk::Function, else_thunk::Function)
    cond_val = TracedUtils.get_mlir_data(cond)
    snap = TracedUtils.get_mlir_data(rng.seed)

    trace_branch =
        (block, thunk) -> begin
            MLIR.IR.activate(block)
            local val
            try
                val = thunk()
                MLIR.Dialects.impulse.yield(
                    MLIR.IR.Value[
                        TracedUtils.get_mlir_data(rng.seed),
                        TracedUtils.get_mlir_data(val),
                    ],
                )
            finally
                MLIR.IR.deactivate(block)
            end
            return val
        end

    then_block = MLIR.IR.Block()
    then_region = MLIR.IR.Region()
    push!(then_region, then_block)
    then_val = trace_branch(then_block, then_thunk)

    TracedUtils.set_mlir_data!(rng.seed, snap)

    else_block = MLIR.IR.Block()
    else_region = MLIR.IR.Region()
    push!(else_region, else_block)
    trace_branch(else_block, else_thunk)

    result_types = [MLIR.IR.type(snap), MLIR.IR.type(TracedUtils.get_mlir_data(then_val))]

    if_op = MLIR.Dialects.impulse.if_(
        cond_val; results=result_types, trueBranch=then_region, falseBranch=else_region
    )

    TracedUtils.set_mlir_data!(rng.seed, MLIR.IR.result(if_op, 1))
    return _if_wrap(then_val, MLIR.IR.result(if_op, 2))
end

function untraced_call(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    args_with_rng = (rng, args...)

    (; f_name, mlir_caller_args, mlir_result_types, traced_result, linear_results, fnwrapped, argprefix, resprefix) = process_probprog_function(
        f, args_with_rng, "untraced_call"
    )

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)

    call_op = MLIR.Dialects.impulse.untracedCall(
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
        ScopedValues.with(TRACING_TRACE => tt) do
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

    simulate_op = MLIR.Dialects.impulse.simulate(
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

    compiled_fn = ScopedValues.with(TRACING_TRACE => tt) do
        @compile optimize = :probprog simulate(rng, f, args...)
    end
    trace_tensor, weight_val, traced_result = compiled_fn(rng, f, args...)

    retval = traced_result[2:end]

    trace = unflatten_trace(trace_tensor, weight_val, tt.entries, retval)
    return trace, trace.weight
end

function flatten_constraint(constraint::Constraint)
    addresses = extract_addresses(constraint)
    flat = Float64[]
    for addr in addresses
        append!(flat, vec(constraint[addr]))
    end
    return to_rarray(reshape(flat, 1, :))
end

# Gen-like helper function.
function generate_(
    rng::AbstractRNG, constraint::Constraint, f::Function, args::Vararg{Any,Nargs}
) where {Nargs}
    tt = TracedTrace()
    constrained_addresses = extract_addresses(constraint)
    constraint_tensor = flatten_constraint(constraint)

    compiled_fn = ScopedValues.with(TRACING_TRACE => tt) do
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
        ScopedValues.with(TRACING_TRACE => tt) do
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
                MLIR.IR.Attribute(
                    MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), sym_addr)
                ),
            )
        end
        push!(constrained_addresses_attr, MLIR.IR.Attribute(address_attr))
    end

    selection_attr = build_selection_attr(tt)
    pos_size = tt.position_size

    trace_type = MLIR.IR.TensorType([1, pos_size], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    constraint_mlir = TracedUtils.get_mlir_data(constraint_tensor)

    generate_op = MLIR.Dialects.impulse.generate(
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
