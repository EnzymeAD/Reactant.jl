module ProbProg

using ..Reactant:
    MLIR,
    TracedUtils,
    AbstractConcreteArray,
    AbstractConcreteNumber,
    AbstractRNG,
    TracedRArray
using ..Compiler: @jit
using Enzyme

mutable struct ProbProgTrace
    choices::Dict{Symbol,Any}
    retval::Any
    weight::Any

    function ProbProgTrace()
        return new(Dict{Symbol,Any}(), nothing, nothing)
    end
end

function addSampleToTraceLowered(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr::Ptr{Any},
    num_dims_ptr::Ptr{Int64},
    shape_array_ptr::Ptr{Int64},
    datatype_width_ptr::Ptr{Int64},
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))

    num_dims = unsafe_load(num_dims_ptr)
    shape_array = unsafe_wrap(Array, shape_array_ptr, num_dims)
    datatype_width = unsafe_load(datatype_width_ptr)

    julia_type = if datatype_width == 32
        Float32
    elseif datatype_width == 64
        Float64
    elseif datatype_width == 1
        Bool
    else
        @ccall printf("Unsupported datatype width: %d\n"::Cstring, datatype_width::Cint)::Cvoid
        return nothing
    end

    typed_ptr = Ptr{julia_type}(sample_ptr)
    if num_dims == 0
        trace.choices[symbol] = unsafe_load(typed_ptr)
    else
        trace.choices[symbol] = copy(unsafe_wrap(Array, typed_ptr, Tuple(shape_array)))
    end

    return nothing
end

function __init__()
    add_sample_to_trace_ptr = @cfunction(
        addSampleToTraceLowered,
        Cvoid,
        (Ptr{Ptr{Any}}, Ptr{Ptr{Any}}, Ptr{Any}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_sample_to_trace::Cstring, add_sample_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    return nothing
end

function sample(
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    logpdf::Union{Nothing,Function}=nothing,
) where {Nargs}
    argprefix::Symbol = gensym("samplearg")
    resprefix::Symbol = gensym("sampleresult")
    resargprefix::Symbol = gensym("sampleresarg")

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
    (; result, linear_args, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            idx -= fnwrap ? 1 : 0
            TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
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

    # (out_idx1, in_idx1, out_idx2, in_idx2, ...)
    alias_pairs = Int64[]
    for (out_idx, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, argprefix)
            in_idx = nothing
            for (i, arg) in enumerate(linear_args)
                if TracedUtils.has_idx(arg, argprefix) &&
                    TracedUtils.get_idx(arg, argprefix) == TracedUtils.get_idx(res, argprefix)
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
        logpdf_args = (example_sample,)
        if !isempty(args) && args[1] isa AbstractRNG
            logpdf_args = (example_sample, Base.tail(args)...)  # TODO: kwargs?
        end

        logpdf_mlir = invokelatest(
            TracedUtils.make_mlir_fn,
            logpdf,
            logpdf_args,
            (),
            string(logpdf),
            false;
            do_transpose=false,
            args_in_result=:all,
        )

        logpdf_sym = TracedUtils.get_attribute_by_name(logpdf_mlir.f, "sym_name")
        logpdf_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(logpdf_sym))
    end

    sample_op = MLIR.Dialects.enzyme.sample(
        batch_inputs;
        outputs=out_tys,
        fn=fn_attr,
        logpdf=logpdf_attr,
        symbol=symbol_addr,
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

    return result
end

function generate(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    trace = ProbProgTrace()

    weight, res = @jit optimize = :probprog generate_internal(f, args...; trace)

    trace.retval = res isa AbstractConcreteArray ? Array(res) : res
    trace.weight = Array(weight)[1]

    return trace, trace.weight
end

function generate_internal(
    f::Function, args::Vararg{Any,Nargs}; trace::ProbProgTrace
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

    f_out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]
    out_tys = [MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64)); f_out_tys]
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    trace_addr = reinterpret(UInt64, pointer_from_objref(trace))

    # Output: (weight, f's outputs...)
    gen_op = MLIR.Dialects.enzyme.generate(
        batch_inputs; outputs=out_tys, fn=fname, trace=trace_addr
    )

    weight = TracedRArray(MLIR.IR.result(gen_op, 1))

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(gen_op, i + 1)  # to skip weight
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

    return weight, result
end

function simulate(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    trace = ProbProgTrace()

    res = @jit optimize = :probprog sync = true simulate_internal(f, args...; trace)

    trace.retval = res isa AbstractConcreteArray ? Array(res) : res

    return trace
end

function simulate_internal(
    f::Function, args::Vararg{Any,Nargs}; trace::ProbProgTrace
) where {Nargs}
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
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    trace_addr = reinterpret(UInt64, pointer_from_objref(trace))

    simulate_op = MLIR.Dialects.enzyme.simulate(
        batch_inputs; outputs=out_tys, fn=fname, trace=trace_addr
    )

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(simulate_op, i)
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

    return result
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

end
