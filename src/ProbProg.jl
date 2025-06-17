module ProbProg

using ..Reactant: MLIR, TracedUtils, AbstractConcreteArray
using Enzyme

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
        trace[symbol] = unsafe_load(typed_ptr)
    else
        trace[symbol] = copy(unsafe_wrap(Array, typed_ptr, Tuple(shape_array)))
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

@noinline function sample!(
    f::Function, args::Vararg{Any,Nargs}; symbol::Symbol=gensym("sample")
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

    traced_output_indices = Int[]
    for (i, res) in enumerate(linear_results)
        if TracedUtils.has_idx(res, resprefix)
            push!(traced_output_indices, i - 1)
        end
    end

    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))

    sample_op = MLIR.Dialects.enzyme.sample(
        batch_inputs;
        outputs=out_tys,
        fn=fn_attr,
        symbol=symbol_addr,
        traced_output_indices=traced_output_indices,
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

@noinline function generate!(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
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

    gen_op = MLIR.Dialects.enzyme.generate(batch_inputs; outputs=out_tys, fn=fname)

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(gen_op, i)
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

@noinline function simulate!(
    f::Function, args::Vararg{Any,Nargs}; trace::Dict{Symbol,Any}
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

function create_trace()
    return Dict{Symbol,Any}()
end

function print_trace(trace::Dict{Symbol,Any})
    println("### Probabilistic Program Trace ###")
    for (symbol, sample) in trace
        println("  $symbol:")
        println("    Sample: $(sample)")
    end
    return println("### End of Trace ###")
end
end
