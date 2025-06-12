module ProbProg

using ..Reactant: MLIR, TracedUtils, AbstractConcreteArray
using Enzyme

struct SampleMetadata
    shape::NTuple{N,Int} where {N}
    element_type::Type
    is_scalar::Bool

    function SampleMetadata(
        shape::NTuple{N,Int}, element_type::Type, is_scalar::Bool
    ) where {N}
        return new(shape, element_type, is_scalar)
    end
end

const SAMPLE_METADATA_CACHE = Dict{Symbol,SampleMetadata}()

function createTrace()
    return Dict{Symbol,Any}(:_integrity_check => 0x123456789abcdef)
end

function addSampleToTraceLowered(
    trace_ptr_ptr::Ptr{Ptr{Cvoid}}, symbol_ptr_ptr::Ptr{Ptr{Cvoid}}, sample_ptr::Ptr{Cvoid}
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))

    @assert haskey(SAMPLE_METADATA_CACHE, symbol) "Symbol $symbol not found in metadata cache"

    metadata = SAMPLE_METADATA_CACHE[symbol]
    shape = metadata.shape
    element_type = metadata.element_type
    is_scalar = metadata.is_scalar

    if is_scalar
        trace[symbol] = unsafe_load(reinterpret(Ptr{element_type}, sample_ptr))
    else
        trace[symbol] = copy(
            reshape(
                unsafe_wrap(
                    Array{element_type},
                    reinterpret(Ptr{element_type}, sample_ptr),
                    prod(shape),
                ),
                shape,
            ),
        )
    end

    return nothing
end

function __init__()
    add_sample_to_trace_ptr = @cfunction(
        addSampleToTraceLowered, Cvoid, (Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Cvoid})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_sample_to_trace::Cstring, add_sample_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    return nothing
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
        for path in res.paths
            isempty(path) && continue
            if path[1] == resprefix
                TracedUtils.set!(result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if idx == 1 && fnwrap
                    TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrap
                        idx -= 1
                    end
                    TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
    end

    return result
end

@noinline function sample!(
    f::Function,
    args::Vararg{Any,Nargs};
    symbol::Symbol=gensym("sample"),
    trace::Union{Dict,Nothing}=nothing,
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
        args_in_result=:all,
        do_transpose=false,  # TODO: double check transpose
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

    if !isempty(linear_results)
        sample_result = linear_results[1] # TODO: consider multiple results
        sample_mlir_data = TracedUtils.get_mlir_data(sample_result)
        @assert sample_mlir_data isa MLIR.IR.Value "Sample $sample_result is not a MLIR.IR.Value"

        sample_type = MLIR.IR.type(sample_mlir_data)
        sample_shape = size(sample_type)
        sample_element_type = MLIR.IR.julia_type(eltype(sample_type))

        SAMPLE_METADATA_CACHE[symbol] = SampleMetadata(
            sample_shape, sample_element_type, length(sample_shape) == 0
        )
    end

    symbol_addr = reinterpret(UInt64, pointer_from_objref(symbol))

    sample_op = MLIR.Dialects.enzyme.sample(
        batch_inputs; outputs=out_tys, fn=fn_attr, symbol=symbol_addr
    )

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(sample_op, i)
        if TracedUtils.has_idx(res, resprefix)
            path = TracedUtils.get_idx(res, resprefix)
            TracedUtils.set!(result, path[2:end], TracedUtils.transpose_val(resv))
        elseif TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if idx == 1 && fnwrap
                TracedUtils.set!(f, path[3:end], TracedUtils.transpose_val(resv))
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], TracedUtils.transpose_val(resv))
            end
        else
            TracedUtils.set!(res, (), TracedUtils.transpose_val(resv))
        end
    end

    return result
end

@noinline function simulate!(
    f::Function, args::Vararg{Any,Nargs}; trace::Dict
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
            TracedUtils.set!(result, path[2:end], TracedUtils.transpose_val(resv))
        elseif TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if idx == 1 && fnwrap
                TracedUtils.set!(f, path[3:end], TracedUtils.transpose_val(resv))
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], TracedUtils.transpose_val(resv))
            end
        else
            TracedUtils.set!(res, (), TracedUtils.transpose_val(resv))
        end
    end

    return trace, result
end

function print_trace(trace::Dict)
    println("Probabilistic Program Trace:")
    for (symbol, sample) in trace
        symbol == :_integrity_check && continue
        metadata = SAMPLE_METADATA_CACHE[symbol]

        println("  $symbol:")
        println("    Sample: $(sample)")
        println("    Shape: $(metadata.shape)")
        println("    Element Type: $(metadata.element_type)")
    end
end

function clear_sample_metadata_cache!()
    empty!(SAMPLE_METADATA_CACHE)
    return nothing
end

end
