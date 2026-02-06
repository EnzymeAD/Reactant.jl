using ScopedValues: ScopedValues, ScopedValue
using ..Reactant:
    MLIR,
    TracedUtils,
    Compiler,
    OrderedIdDict,
    TracedToTypes,
    TracedType,
    TracedTrack,
    TracedSetPath

import ..Reactant: make_tracer

const TRACING_TRACE = ScopedValue{Union{Nothing,TracedTrace}}(nothing)

function process_probprog_function(f, args, op_name, with_rng=true)
    seen = OrderedIdDict()
    cache_key = []
    make_tracer(seen, (f, args...), cache_key, TracedToTypes)
    cache = Compiler.callcache()

    collecting_metadata = TRACING_TRACE[] !== nothing

    if !collecting_metadata && haskey(cache, cache_key)
        (; f_name, mlir_result_types, traced_result, mutated_args, linear_results, fnwrapped, argprefix, resprefix, resargprefix) = cache[cache_key]
    else
        f_name = String(gensym(Symbol(f)))
        argprefix::Symbol = gensym(op_name * "arg")
        resprefix::Symbol = gensym(op_name * "result")
        resargprefix::Symbol = gensym(op_name * "resarg")

        wrapper_fn = if !with_rng
            f
        else
            (all_args...) -> begin
                res = f(all_args...)
                (all_args[1], (res isa Tuple ? res : (res,))...)
            end
        end

        temp = TracedUtils.make_mlir_fn(
            wrapper_fn,
            args,
            (),
            f_name,
            false;
            do_transpose=false,
            args_in_result=:result,
            argprefix,
            resprefix,
            resargprefix,
        )

        (; traced_result, ret, mutated_args, linear_results, fnwrapped) = temp
        mlir_result_types = [
            MLIR.IR.type(MLIR.IR.operand(ret, i)) for i in 1:MLIR.IR.noperands(ret)
        ]
        if !collecting_metadata
            cache[cache_key] = (;
                f_name,
                mlir_result_types,
                traced_result,
                mutated_args,
                linear_results,
                fnwrapped,
                argprefix,
                resprefix,
                resargprefix,
            )
        end
    end

    seen_cache = OrderedIdDict()
    make_tracer(seen_cache, fnwrapped ? (f, args) : args, (), TracedTrack; toscalar=false)
    linear_args = []
    mlir_caller_args = MLIR.IR.Value[]
    for (_, v) in seen_cache
        v isa TracedType || continue
        push!(linear_args, v)
        push!(mlir_caller_args, v.mlir_data)
        v.paths = v.paths[1:(end - 1)]
    end

    return (;
        f_name,
        linear_args,
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
        resargprefix,
    )
end

function process_probprog_outputs(
    op,
    linear_results,
    traced_result,
    f,
    args,
    fnwrapped,
    resprefix,
    argprefix,
    offset=0,
    rng_only=false,
)
    seen_results = OrderedIdDict()
    traced_result = make_tracer(
        seen_results, traced_result, (), TracedSetPath; toscalar=false
    )

    num_to_process = rng_only ? 1 : length(linear_results)

    for i in 1:num_to_process
        res = linear_results[i]
        resv = MLIR.IR.result(op, i + offset)

        for path in res.paths
            if length(path) == 0
                continue
            end
            if path[1] == resprefix
                TracedUtils.set!(traced_result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if fnwrapped && idx == 2
                    TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrapped && idx > 2
                        idx -= 1
                    end
                    TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
    end

    return traced_result
end

function build_selection_attr(trace::TracedTrace)
    selection = MLIR.IR.Attribute[]
    for entry in trace.entries
        addr_path = [entry.parent_path..., entry.symbol]
        addr_attrs = [
            (@ccall MLIR.API.mlir_c.enzymeSymbolAttrGet(
                MLIR.IR.current_context()::MLIR.API.MlirContext,
                reinterpret(UInt64, pointer_from_objref(sym))::UInt64,
            )::MLIR.IR.Attribute) for sym in addr_path
        ]
        push!(selection, MLIR.IR.Attribute(addr_attrs))
    end
    return MLIR.IR.Attribute(selection)
end

function filter_entries_by_selection(entries::Vector{TraceEntry}, selection::Selection)
    filtered = TraceEntry[]
    new_offset = 0
    for address in selection
        for entry in entries
            entry_path = [entry.parent_path..., entry.symbol]
            if entry_path == collect(address.path)
                push!(
                    filtered,
                    TraceEntry(
                        entry.symbol,
                        entry.shape,
                        entry.num_elements,
                        new_offset,
                        entry.parent_path,
                    ),
                )
                new_offset += entry.num_elements
                break
            end
        end
    end
    return filtered
end

function unflatten_trace(trace_tensor, weight, entries::Vector{TraceEntry}, retval)
    result = Trace()
    result.weight =
        weight isa AbstractArray ? Float64(only(Array(weight))) : Float64(weight)
    result.retval = retval

    trace_arr = Array(trace_tensor)
    num_samples = size(trace_arr, 1)

    for entry in entries
        start_col = entry.offset + 1
        stop_col = start_col + entry.num_elements - 1

        raw = trace_arr[:, start_col:stop_col]
        if entry.shape == ()
            value = vec(raw)
        else
            value = reshape(raw, (num_samples, entry.shape...))
        end

        target = result
        for psym in entry.parent_path
            if !haskey(target.subtraces, psym)
                target.subtraces[psym] = Trace()
            end
            target = target.subtraces[psym]
        end
        target.choices[entry.symbol] = value
    end
    return result
end
