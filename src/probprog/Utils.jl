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

function with_trace(f, tt::TracedTrace=TracedTrace())
    result = ScopedValues.with(f, TRACING_TRACE => tt)
    return result, tt
end

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

function trace_logpdf_function(logdensity_fn, sample_pos, args)
    logpdf_name = String(gensym(Symbol(logdensity_fn)))

    seen = OrderedIdDict()
    for (i, arg) in enumerate(args)
        make_tracer(seen, arg, (:_logpdf_trace, i), TracedTrack; toscalar=false)
    end
    traced_vals = TracedType[]
    for (_, v) in seen
        v isa TracedType || continue
        push!(traced_vals, v)
        v.paths = v.paths[1:(end - 1)]
    end

    saved_paths = Tuple[v.paths for v in traced_vals]
    TracedUtils.make_mlir_fn(
        logdensity_fn,
        (sample_pos, args...),
        (),
        logpdf_name,
        false;
        do_transpose=false,
        args_in_result=:result,
    )
    for (i, v) in enumerate(traced_vals)
        v.paths = saved_paths[i]
    end

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(logpdf_name)
    extra_mlir_data = MLIR.IR.Value[v.mlir_data for v in traced_vals]
    return fn_attr, extra_mlir_data
end

function build_selection_attr(trace::TracedTrace)
    selection = MLIR.IR.Attribute[]
    for entry in trace.entries
        addr_path = [entry.parent_path..., entry.symbol]
        addr_attrs = [
            MLIR.IR.Attribute(
                MLIR.API.enzymeSymbolAttrGet(
                    MLIR.IR.current_context(), reinterpret(UInt64, pointer_from_objref(sym))
                ),
            ) for sym in addr_path
        ]
        push!(selection, MLIR.IR.Attribute(addr_attrs))
    end
    return MLIR.IR.Attribute(selection)
end

"""
    position_layout(selection) -> Selection

The single ordering convention for everything positional in an `impulse.infer`
run: the unconstrained position vector the sampler integrates, the mass matrix
and `initial_position` that index it, the per-site `supports` offsets, and the
collected-samples buffer's columns. It is the order of the `selection` attribute
handed to the op, so the C++ layout builders (`collectSupportInfoForSelection`
and `collectSupportInfoForRegion`, both of which walk `selection`) and this
frontend derive the layout from one place.

It is deliberately NOT the caller's `ProbProg.select` argument order. Since
upstream e0cb37d73 ("Respect order of parameter specification", #3059) that
argument order is a user-facing choice governing the OUTPUT column order, and
letting it also drive the integration layout made the sampler's coordinates
model-author-dependent. In particular it broke pointwise agreement with
NumPyro, which flattens its momentum with `ravel_pytree` over the site dict and
`jax.tree_util` sorts dict keys, so NumPyro's layout is sorted-by-address. A
single flat momentum draw only lands on the same coordinates in both systems
when our layout is sorted too.

The two orders are reconciled in exactly one place: `_relayout_to_selection`
permutes the returned trace columns from this layout back to the caller's
`select` order, so `#3059`'s output-column contract still holds.
"""
# Not by string.(path): String isless recurses under Reactant tracing until stack overflow.
# Symbol order matches NumPyro's name-sorted site order.
position_layout(selection::Selection) = Selection(sort(collect(selection); by=a -> a.path))

"""
    layout_column_ranges(entries, layout, selection) -> Vector{UnitRange{Int}}

Constrained column ranges of the collected-samples buffer, one per address in
`selection` order, addressed in `position_layout` order. Feeding these to
`_relayout_to_selection` turns a layout-ordered buffer into a selection-ordered
one. Returns an empty vector when the two orders already agree.
"""
function layout_column_ranges(
    entries::Vector{TraceEntry}, layout::Selection, selection::Selection
)
    collect(layout) == collect(selection) && return UnitRange{Int}[]
    width = Dict{Address,Int}()
    for address in layout
        for entry in entries
            if [entry.parent_path..., entry.symbol] == collect(address.path)
                width[address] = entry.num_elements
                break
            end
        end
    end
    span = Dict{Address,UnitRange{Int}}()
    offset = 0
    for address in layout
        n = get(width, address, 0)
        span[address] = (offset + 1):(offset + n)
        offset += n
    end
    return UnitRange{Int}[span[address] for address in selection]
end

function _relayout_to_selection(x, ranges::Vector{UnitRange{Int}})
    isempty(ranges) && return x
    sz = size(x)
    nd = length(sz)
    pieces = map(ranges) do r
        lo = ones(Int, nd)
        hi = collect(Int, sz)
        lo[nd] = first(r)
        hi[nd] = last(r)
        return Ops.slice(x, lo, hi)
    end
    length(pieces) == 1 && return only(pieces)
    return Ops.concatenate(pieces, nd)
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
                        entry.support,
                        entry.event_size,
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
