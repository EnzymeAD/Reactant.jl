module ProbProg

using ..Reactant:
    MLIR,
    TracedUtils,
    AbstractConcreteArray,
    AbstractConcreteNumber,
    AbstractRNG,
    TracedRArray,
    TracedRNumber,
    ConcreteRNumber,
    Ops
using ..Compiler: @jit, @compile
using Enzyme
using Base: ReentrantLock

mutable struct ProbProgTrace
    fn::Union{Nothing,Function}
    args::Union{Nothing,Tuple}
    choices::Dict{Symbol,Any}
    retval::Any
    weight::Any
    subtraces::Dict{Symbol,Any}

    function ProbProgTrace(fn::Function, args::Tuple)
        return new(fn, args, Dict{Symbol,Any}(), nothing, nothing, Dict{Symbol,Any}())
    end

    function ProbProgTrace()
        return new(nothing, (), Dict{Symbol,Any}(), nothing, nothing, Dict{Symbol,Any}())
    end
end

const Constraint = Dict{Symbol,Any}

const _trace_ref_lock = ReentrantLock()
const _trace_refs = Vector{Any}()

function _keepalive!(tr::ProbProgTrace)
    lock(_trace_ref_lock)
    try
        push!(_trace_refs, tr)
    finally
        unlock(_trace_ref_lock)
    end
    return tr
end

function initTrace(trace_ptr_ptr::Ptr{Ptr{Any}})
    tr = ProbProgTrace()
    _keepalive!(tr)

    unsafe_store!(trace_ptr_ptr, pointer_from_objref(tr))
    return nothing
end

function addSampleToTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr_array::Ptr{Ptr{Any}},
    num_outputs_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_outputs = unsafe_load(num_outputs_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_outputs)
    width_array = unsafe_wrap(Array, width_array, num_outputs)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_outputs)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_outputs)

    tostore = Any[]
    for i in 1:num_outputs
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        sample_ptr = sample_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %lld\n"::Cstring, width::Int64
            )::Cvoid
            return nothing
        end

        if ndims == 0
            val = unsafe_load(Ptr{julia_type}(sample_ptr))
            push!(tostore, val)
        else
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            push!(
                tostore, copy(unsafe_wrap(Array, Ptr{julia_type}(sample_ptr), Tuple(shape)))
            )
        end
    end

    trace.choices[symbol] = tuple(tostore...)

    return nothing
end

function addSubtrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    subtrace_ptr_ptr::Ptr{Ptr{Any}},
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    subtrace = unsafe_pointer_to_objref(unsafe_load(subtrace_ptr_ptr))::ProbProgTrace

    trace.subtraces[symbol] = subtrace

    return nothing
end

function addWeightToTrace(trace_ptr_ptr::Ptr{Ptr{Any}}, weight_ptr::Ptr{Any})
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    trace.weight = unsafe_load(Ptr{Float64}(weight_ptr))
    return nothing
end

function addRetvalToTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    retval_ptr_array::Ptr{Ptr{Any}},
    num_results_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace

    num_results = unsafe_load(num_results_ptr)

    if num_results == 0
        return nothing
    end

    ndims_array = unsafe_wrap(Array, ndims_array, num_results)
    width_array = unsafe_wrap(Array, width_array, num_results)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_results)
    retval_ptr_array = unsafe_wrap(Array, retval_ptr_array, num_results)

    vals = Any[]
    for i in 1:num_results
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        retval_ptr = retval_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %lld\n"::Cstring, width::Int64
            )::Cvoid
            return nothing
        end

        if ndims == 0
            push!(vals, unsafe_load(Ptr{julia_type}(retval_ptr)))
        else
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            push!(vals, copy(unsafe_wrap(Array, Ptr{julia_type}(retval_ptr), Tuple(shape))))
        end
    end

    trace.retval = length(vals) == 1 ? vals[1] : vals
    return nothing
end

function getSampleFromConstraint(
    constraint_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr_array::Ptr{Ptr{Any}},
    num_samples_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    constraint = unsafe_pointer_to_objref(unsafe_load(constraint_ptr_ptr))::Constraint
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_samples = unsafe_load(num_samples_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_samples)
    width_array = unsafe_wrap(Array, width_array, num_samples)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_samples)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_samples)

    tostore = get(constraint, symbol, nothing)

    for i in 1:num_samples
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        sample_ptr = sample_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %zd\n"::Cstring, width::Csize_t
            )::Cvoid
            return nothing
        end

        if julia_type != eltype(tostore[i])
            @ccall printf(
                "Type mismatch in constrained sample: %s != %s\n"::Cstring,
                string(julia_type)::Cstring,
                string(eltype(tostore[i]))::Cstring,
            )::Cvoid
            return nothing
        end

        if ndims == 0
            unsafe_store!(Ptr{julia_type}(sample_ptr), tostore[i])
        else
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            dest = unsafe_wrap(Array, Ptr{julia_type}(sample_ptr), Tuple(shape))

            if size(dest) != size(tostore[i])
                if length(size(dest)) != length(size(tostore[i]))
                    @ccall printf(
                        "Shape size mismatch in constrained sample: %zd != %zd\n"::Cstring,
                        length(size(dest))::Csize_t,
                        length(size(tostore[i]))::Csize_t,
                    )::Cvoid
                    return nothing
                end
                for i in 1:length(size(dest))
                    d = size(dest)[i]
                    t = size(tostore[i])[i]
                    if d != t
                        @ccall printf(
                            "Shape mismatch in `%zd`th dimension of constrained sample: %zd != %zd\n"::Cstring,
                            i::Csize_t,
                            size(dest)[i]::Csize_t,
                            size(tostore[i])[i]::Csize_t,
                        )::Cvoid
                        return nothing
                    end
                end
            end

            dest .= tostore[i]
        end
    end

    return nothing
end

function __init__()
    init_trace_ptr = @cfunction(initTrace, Cvoid, (Ptr{Ptr{Any}},))
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_init_trace::Cstring, init_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_sample_to_trace_ptr = @cfunction(
        addSampleToTrace,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        )
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_sample_to_trace::Cstring, add_sample_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_subtrace_ptr = @cfunction(
        addSubtrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Ptr{Any}}, Ptr{Ptr{Any}})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_subtrace::Cstring, add_subtrace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_weight_to_trace_ptr = @cfunction(addWeightToTrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Any}))
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_weight_to_trace::Cstring, add_weight_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_retval_to_trace_ptr = @cfunction(
        addRetvalToTrace,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        ),
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_retval_to_trace::Cstring, add_retval_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    get_sample_from_constraint_ptr = @cfunction(
        getSampleFromConstraint,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        )
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_sample_from_constraint::Cstring,
        get_sample_from_constraint_ptr::Ptr{Cvoid},
    )::Cvoid

    return nothing
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
    argprefix::Symbol = gensym("samplearg")
    resprefix::Symbol = gensym("sampleresult")
    resargprefix::Symbol = gensym("sampleresarg")

    wrapper_fn = (all_args...) -> begin
        res = f(all_args...)
        (all_args[1], (res isa Tuple ? res : (res,))...)
    end

    args = (rng, args...)

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
    (; result, linear_args, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

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

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    sym = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(sym))

    # Specify which outputs to add to the trace.
    traced_output_indices = Int[]
    for (i, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, resprefix)
            push!(traced_output_indices, i - 1)
        end
    end

    # Specify which inputs to pass to logpdf.
    traced_input_indices = Int[]
    for (i, a) in enumerate(linear_args)
        idx, _ = TracedUtils.get_argidx(a, argprefix)
        if fnwrap && idx == 1  # TODO: add test for fnwrap
            continue
        end

        if fnwrap
            idx -= 1
        end

        if !(args[idx] isa AbstractRNG)
            push!(traced_input_indices, i - 1)
        end
    end

    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))
    symbol_attr = @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
        MLIR.IR.context()::MLIR.API.MlirContext, symbol_addr::UInt64
    )::MLIR.IR.Attribute

    # (out_idx1, in_idx1, out_idx2, in_idx2, ...)
    alias_pairs = Int64[]
    for (out_idx, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, argprefix)
            in_idx = nothing
            for (i, arg) in enumerate(linear_args)
                if TracedUtils.has_idx(arg, argprefix) &&
                    TracedUtils.get_idx(arg, argprefix) ==
                   TracedUtils.get_idx(res, argprefix)
                    in_idx = i - 1
                    break
                end
            end
            @assert in_idx !== nothing "Unable to find operand for aliased result"
            push!(alias_pairs, out_idx - 1)
            push!(alias_pairs, in_idx)
        end
    end
    alias_attr = MLIR.IR.DenseArrayAttribute(alias_pairs)

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
        traced_input_indices=traced_input_indices,
        traced_output_indices=traced_output_indices,
        alias_map=alias_attr,
        name=Base.String(symbol),
    )

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(sample_op, i)

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

    return result
end

function call(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    res = @jit optimize = :probprog call_internal(rng, f, args...)

    @assert res isa Tuple && length(res) >= 1 && res[1] isa AbstractRNG "Expected first result to be RNG"

    res = map(res[2:end]) do r
        r isa AbstractConcreteArray ? Array(r) : r
    end

    @show res

    return length(res) == 1 ? res[1] : res
end

function call_internal(rng::AbstractRNG, f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    argprefix::Symbol = gensym("callarg")
    resprefix::Symbol = gensym("callresult")
    resargprefix::Symbol = gensym("callresarg")

    wrapper_fn = (all_args...) -> begin
        res = f(all_args...)
        (all_args[1], (res isa Tuple ? res : (res,))...)
    end

    args = (rng, args...)

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
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    @show length(linear_results), linear_results

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(inputs, args[idx], path[3:end])
        end
    end

    call_op = MLIR.Dialects.enzyme.untracedCall(inputs; outputs=out_tys, fn=fn_attr)

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(call_op, i)
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

    return result
end

function simulate(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    old_gc_state = GC.enable(false)

    trace = nothing
    weight = nothing
    res = nothing

    try
        trace, weight, res = @jit optimize = :probprog simulate_internal(f, args...)
    finally
        GC.enable(old_gc_state)
    end

    trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace)[1]))

    return trace, trace.weight
end

function simulate_internal(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    argprefix::Symbol = gensym("simulatearg")
    resprefix::Symbol = gensym("simulateresult")
    resargprefix::Symbol = gensym("simulateresarg")

    mlir_fn_res = invokelatest(
        TracedUtils.make_mlir_fn,
        f,
        args,
        (),
        string(f),
        false;
        do_transpose=false,
        args_in_result=:all,
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    # Specify which outputs to add to the trace.
    traced_output_indices = Int[]
    for (i, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, resprefix)
            push!(traced_output_indices, i - 1)
        end
    end

    inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(inputs, args[idx], path[3:end])
        end
    end

    trace_ty = @ccall MLIR.API.mlir_c.enzymeTraceTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type
    weight_ty = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    simulate_op = MLIR.Dialects.enzyme.simulate(
        inputs;
        trace=trace_ty,
        weight=weight_ty,
        outputs=out_tys,
        fn=fn_attr,
        traced_output_indices=traced_output_indices,
    )

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(simulate_op, i + 2)
        if TracedUtils.has_idx(res, resprefix)
            path = TracedUtils.get_idx(res, resprefix)
            TracedUtils.set!(result, path[2:end], resv)
        elseif TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if idx == 1 && fnwrap
                TracedUtils.set!(f, path[3:end], resv)
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], resv)
            end
        else
            TracedUtils.set!(res, (), resv)
        end
    end

    trace = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(simulate_op, 1)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))],
        ),
        1,
    )

    weight = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(simulate_op, 2)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))],
        ),
        1,
    )

    trace = TracedRArray{UInt64,0}((), trace, ())
    weight = TracedRArray{Float64,0}((), weight, ())

    return trace, weight, result
end

function generate(
    f::Function, args::Vararg{Any,Nargs}; constraint::Constraint=Dict{Symbol,Any}()
) where {Nargs}
    trace = nothing
    weight = nothing
    res = nothing

    constraint_ptr = ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(constraint)))
    constrained_symbols = collect(keys(constraint))

    function wrapper_fn(constraint_ptr, args...)
        return generate_internal(f, args...; constraint_ptr, constrained_symbols)
    end

    compiled_fn = @compile optimize = :probprog wrapper_fn(constraint_ptr, args...)

    old_gc_state = GC.enable(false)
    try
        trace, weight, res = compiled_fn(constraint_ptr, args...)
    finally
        GC.enable(old_gc_state)
    end

    trace = unsafe_pointer_to_objref(Ptr{Any}(Array(trace)[1]))

    return trace, trace.weight
end

function generate_internal(
    f::Function,
    args::Vararg{Any,Nargs};
    constraint_ptr::TracedRNumber,
    constrained_symbols::Vector{Symbol},
) where {Nargs}
    argprefix::Symbol = gensym("generatearg")
    resprefix::Symbol = gensym("generateresult")
    resargprefix::Symbol = gensym("generateresarg")

    mlir_fn_res = invokelatest(
        TracedUtils.make_mlir_fn,
        f,
        args,
        (),
        string(f),
        false;
        do_transpose=false,
        args_in_result=:all,
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    # Specify which outputs to add to the trace.
    traced_output_indices = Int[]
    for (i, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, resprefix)
            push!(traced_output_indices, i - 1)
        end
    end

    inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(inputs, args[idx], path[3:end])
        end
    end

    constraint_ty = @ccall MLIR.API.mlir_c.enzymeConstraintTypeGet(
        MLIR.IR.context()::MLIR.API.MlirContext
    )::MLIR.IR.Type

    constraint_val = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [TracedUtils.get_mlir_data(constraint_ptr)]; outputs=[constraint_ty]
        ),
        1,
    )

    constrained_symbols_attr = MLIR.IR.Attribute[]
    for sym in constrained_symbols
        addr = reinterpret(UInt64, pointer_from_objref(sym))
        push!(
            constrained_symbols_attr,
            @ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
                MLIR.IR.context()::MLIR.API.MlirContext, addr::UInt64
            )::MLIR.IR.Attribute
        )
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
        constrained_symbols=MLIR.IR.Attribute(constrained_symbols_attr),
        traced_output_indices,
    )

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(generate_op, i + 2)
        if TracedUtils.has_idx(res, resprefix)
            path = TracedUtils.get_idx(res, resprefix)
            TracedUtils.set!(result, path[2:end], resv)
        elseif TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if idx == 1 && fnwrap
                TracedUtils.set!(f, path[3:end], resv)
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], resv)
            end
        else
            TracedUtils.set!(res, (), resv)
        end
    end

    trace = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(generate_op, 1)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(UInt64))],
        ),
        1,
    )

    weight = MLIR.IR.result(
        MLIR.Dialects.builtin.unrealized_conversion_cast(
            [MLIR.IR.result(generate_op, 2)];
            outputs=[MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))],
        ),
        1,
    )

    trace = TracedRArray{UInt64,0}((), trace, ())
    weight = TracedRArray{Float64,0}((), weight, ())

    return trace, weight, result
end

# Reference: https://github.com/probcomp/Gen.jl/blob/91d798f2d2f0c175b1be3dc6daf3a10a8acf5da3/src/choice_map.jl#L104
function _show_pretty(io::IO, trace::ProbProgTrace, pre::Int, vert_bars::Tuple)
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'

    indent_vert = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent = vcat(Char[' ' for _ in 1:pre], Char[PLUS, HORZ, HORZ, ' '])
    indent_last = vcat(Char[' ' for _ in 1:pre], Char[LAST, HORZ, HORZ, ' '])

    for i in vert_bars
        indent_vert[i] = VERT
        indent[i] = VERT
        indent_last[i] = VERT
    end

    indent_vert_str = join(indent_vert)
    indent_str = join(indent)
    indent_last_str = join(indent_last)

    sorted_choices = sort(collect(trace.choices); by=x -> x[1])
    n = length(sorted_choices)

    if trace.retval !== nothing
        n += 1
    end

    if trace.weight !== nothing
        n += 1
    end

    cur = 1

    if trace.retval !== nothing
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "retval : $(trace.retval)\n")
        cur += 1
    end

    if trace.weight !== nothing
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "weight : $(trace.weight)\n")
        cur += 1
    end

    for (key, value) in sorted_choices
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value\n")
        cur += 1
    end

    sorted_subtraces = sort(collect(trace.subtraces); by=x -> x[1])
    n += length(sorted_subtraces)

    for (key, subtrace) in sorted_subtraces
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "subtrace on $(repr(key))\n")
        _show_pretty(
            io, subtrace, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre + 1)
        )
        cur += 1
    end
end

function Base.show(io::IO, ::MIME"text/plain", trace::ProbProgTrace)
    println(io, "ProbProgTrace:")
    if isempty(trace.choices) && trace.retval === nothing && trace.weight === nothing
        println(io, "  (empty)")
    else
        _show_pretty(io, trace, 0, ())
    end
end

function Base.show(io::IO, trace::ProbProgTrace)
    if get(io, :compact, false)
        choices_count = length(trace.choices)
        has_retval = trace.retval !== nothing
        print(io, "ProbProgTrace($(choices_count) choices")
        if has_retval
            print(io, ", retval=$(trace.retval), weight=$(trace.weight)")
        end
        print(io, ")")
    else
        show(io, MIME"text/plain"(), trace)
    end
end

get_choices(trace::ProbProgTrace) = trace.choices

end
