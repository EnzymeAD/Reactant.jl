# Functions within this module and Ops do not get forcibly re-compiled to be within our interpreter.
# This means that replacements, for example, for autodiff/random/kernels/etc do not get applied here when
# within compilation. However, it means these functions are a _lot_ faster to compile.
module TracedUtils

using Adapt: Adapt, WrappedReshapedArray
using ..Reactant:
    Reactant,
    MLIR,
    RNumber,
    TracedRArray,
    TracedRNumber,
    WrappedTracedRArray,
    AnyTracedRArray,
    MissingTracedValue,
    OrderedIdDict,
    ReactantPrimitive,
    Ops
using ReactantCore: MissingTracedValue

materialize_traced_array(x::TracedRArray) = x

materialize_traced_array(x::WrappedTracedRArray) = x[axes(x)...]

function materialize_traced_array(
    x::WrappedReshapedArray{TracedRNumber{T},N,TracedRArray{T,M}}
) where {T,N,M}
    return Ops.reshape(materialize_traced_array(parent(x)), size(x)...)
end

function materialize_traced_array(
    x::PermutedDimsArray{TracedRNumber{T},N,perm,iperm,TracedRArray{T,N}}
) where {T,N,perm,iperm}
    return permutedims(parent(x), perm)
end

get_mlir_data(x::TracedRNumber) = x.mlir_data
set_mlir_data!(x::TracedRNumber, data) = (x.mlir_data = data; return x)
get_paths(x::TracedRNumber) = x.paths
set_paths!(x::TracedRNumber, paths) = (x.paths = paths; return x)

get_mlir_data(x::TracedRArray) = x.mlir_data
get_mlir_data(x::AnyTracedRArray) = get_mlir_data(materialize_traced_array(x))
get_paths(x::TracedRArray) = x.paths
set_paths!(x::TracedRArray, paths) = (x.paths = paths; return x)

get_paths(x::MissingTracedValue) = x.paths
set_paths!(x::MissingTracedValue, paths) = (x.paths = paths; return x)

function set_mlir_data!(x::TracedRArray, data)
    x.mlir_data = data
    return x
end

function set_mlir_data!(
    x::WrappedReshapedArray{TracedRNumber{T},N,TracedRArray{T,M}}, data
) where {T,N,M}
    res_mlir_data = Ops.reshape(TracedRArray{T}(data), size(parent(x))...).mlir_data
    set_mlir_data!(parent(x), res_mlir_data)
    return x
end

function get_ancestor_indices(
    x::WrappedReshapedArray{TracedRNumber{T},N,TracedRArray{T,M}}, indices...
) where {T,N,M}
    cartesian_indices = CartesianIndex.(indices...)
    linear_indices = LinearIndices(size(x))[cartesian_indices]
    parent_cartesian_indices = CartesianIndices(size(parent(x)))[linear_indices]
    return (parent_cartesian_indices,)
end

function set_mlir_data!(
    x::PermutedDimsArray{TracedRNumber{T},N,perm,iperm,TracedRArray{T,N}}, data
) where {T,N,perm,iperm}
    parent(x).mlir_data = permutedims(TracedRArray{T}(data), iperm).mlir_data
    return x
end

function set_mlir_data!(x::AnyTracedRArray{T}, data) where {T}
    ancestor_indices = TracedUtils.get_ancestor_indices(x, axes(x)...)
    setindex!(Reactant.ancestor(x), TracedRArray{T}(data), ancestor_indices...)
    return x
end

get_ancestor_indices(::TracedRArray, indices...) = indices
function get_ancestor_indices(x::WrappedTracedRArray, indices...)
    return get_ancestor_indices(parent(x), Base.reindex(parentindices(x), indices)...)
end

function batch_ty(width, mlirty)
    return MLIR.IR.TensorType([width, size(mlirty)...], eltype(mlirty))
end

function transpose_ty(mlirty)
    return MLIR.IR.TensorType([reverse(size(mlirty))...], eltype(mlirty))
end
function transpose_val(val)
    val_size = size(MLIR.IR.type(val))
    val_size == () && return val
    attr = MLIR.IR.DenseArrayAttribute(Int64[reverse(0:(length(val_size) - 1))...])
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val; permutation=attr), 1)
end

function make_mlir_fn(
    f,
    args,
    kwargs,
    name="main",
    concretein=true;
    batchmode=Reactant.BatchNone,
    return_dialect=:func,
    do_transpose=true,
    no_args_in_result=false,
)
    if sizeof(typeof(f)) != 0 || f isa Base.BroadcastFunction
        return (
            true,
            make_mlir_fn(
                Reactant.apply,
                (f, args...),
                kwargs,
                name,
                concretein;
                batchmode,
                return_dialect,
                do_transpose,
                no_args_in_result,
            )[2:end]...,
        )
    end

    N = length(args)
    seen_args = OrderedIdDict()
    traced_args = Vector{Any}(undef, N)
    for i in 1:N
        @inbounds traced_args[i] = Reactant.make_tracer(
            seen_args,
            args[i],
            (:args, i),
            concretein ? Reactant.ConcreteToTraced : Reactant.TracedSetPath;
            batchmode
        )
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
    end

    in_tys = if batchmode == Reactant.BatchScalar
        [
            MLIR.IR.TensorType((), MLIR.IR.Type(Reactant.unwrapped_eltype(arg))) for arg in linear_args
        ]
    elseif batchmode == Reactant.BatchArray
        error("Not implemented")
    elseif do_transpose
        [transpose_ty(Ops.mlir_type(arg)) for arg in linear_args]
    else
        [Ops.mlir_type(arg) for arg in linear_args]
    end

    sym_visibility = nothing
    if !concretein
        sym_visibility = MLIR.IR.Attribute("private")
    end

    mod = MLIR.IR.mmodule()
    func = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=name * "_tmp",
            function_type=MLIR.IR.FunctionType(in_tys, []),
            body=MLIR.IR.Region(),
        )
    end

    fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
    push!(MLIR.IR.region(func, 1), fnbody)

    @assert MLIR.IR._has_block()

    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues
    MLIR.IR.activate!(fnbody)
    result = try
        for (i, arg) in enumerate(linear_args)
            raw_arg = MLIR.IR.argument(fnbody, i)
            row_maj_arg = do_transpose ? transpose_val(raw_arg) : raw_arg
            set_mlir_data!(arg, row_maj_arg)
        end

        Reactant.call_with_reactant(f, traced_args...)
    finally
        MLIR.IR.deactivate!(fnbody)
    end

    seen_results = OrderedIdDict()

    traced_result = Reactant.make_tracer(
        seen_results,
        result,
        (:result,),
        concretein ? Reactant.TracedTrack : Reactant.TracedSetPath,
    )

    # marks buffers to be donated
    for i in 1:N
        Reactant.make_tracer(
            seen_results,
            traced_args[i],
            concretein ? (:resargs, i) : (),
            Reactant.TracedTrack,
        )
    end

    linear_results = Reactant.TracedType[]
    for (k, v) in seen_results
        v isa Reactant.TracedType || continue
        (no_args_in_result && has_argidx(v)) && continue
        push!(linear_results, v)
    end

    out_tys = [transpose_ty(Ops.mlir_type(arg)) for arg in linear_results]

    MLIR.IR.activate!(fnbody)
    ret = try
        vals = MLIR.IR.Value[]
        for res in linear_results
            col_maj = if res isa MissingTracedValue
                get_mlir_data(broadcast_to_size(false, ()))
            elseif !do_transpose
                get_mlir_data(res)
            elseif do_transpose
                transpose_val(get_mlir_data(res))
            end
            push!(vals, col_maj)
        end
        !no_args_in_result && @assert length(vals) == length(linear_results)

        dialect = getfield(MLIR.Dialects, return_dialect)
        dialect.return_(vals)
    finally
        MLIR.IR.deactivate!(fnbody)
    end

    func2 = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=__lookup_unique_name_in_module(mod, name),
            function_type=MLIR.IR.FunctionType(in_tys, out_tys),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func2, 1), MLIR.IR.region(func, 1))

    MLIR.API.mlirOperationDestroy(func.operation)
    func.operation = MLIR.API.MlirOperation(C_NULL)
    return (
        false,
        func2,
        traced_result,
        result,
        seen_args,
        ret,
        linear_args,
        in_tys,
        linear_results,
    )
end

function __lookup_unique_name_in_module(mod, name)
    new_name = name
    tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
    for i in 0:10000
        new_name = i == 0 ? name : name * "_" * string(i)
        MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, new_name)) && return new_name
    end
    return error("Could not find unique name for $name")
end

function __take_region(compiled_fn)
    region = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(region, MLIR.API.mlirOperationGetRegion(compiled_fn, 0))
    return region
end

function promote_to end

function get_attribute_by_name(operation, name)
    return MLIR.IR.Attribute(MLIR.API.mlirOperationGetAttributeByName(operation, name))
end

function push_val!(ad_inputs, x, path)
    for p in path
        x = Reactant.Compiler.traced_getfield(x, p)
    end
    x = get_mlir_data(x)
    return push!(ad_inputs, x)
end

function get_argidx(x)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == :args
            return path[2]::Int, path
        end
    end
    throw(AssertionError("No path found for $x"))
end

function has_argidx(x)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == :args
            return true
        end
    end
    return false
end

function set!(x, path, tostore; emptypath=false)
    for p in path
        x = Reactant.Compiler.traced_getfield(x, p)
    end

    if tostore isa TracedRArray
        set_mlir_data!(x, tostore.mlir_data)
    else
        set_mlir_data!(x, tostore)
    end

    return emptypath && set_paths!(x, ())
end

function get_residx(x)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == :result
            return path
        end
    end
    throw(AssertionError("No path found $x"))
end

function has_residx(x)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == :result
            return true
        end
    end
    return false
end

function broadcast_to_size(arg::AbstractArray{<:TracedRNumber}, rsize)
    return broadcast_to_size(reshape(Ops.vcat(arg...), size(arg)...), rsize)
end
broadcast_to_size(arg::AbstractArray, rsize) = broadcast_to_size(Ops.constant(arg), rsize)

function broadcast_to_size(arg::Base.RefValue, rsize)
    # XXX: don't we want to expand here to rsize?
    return arg
end

broadcast_to_size(arg::Number, rsize) = Ops.constant(Base.fill(arg, Tuple(rsize)))

function broadcast_to_size(arg::TracedRNumber{T}, rsize) where {T}
    length(rsize) == 0 && return arg
    return broadcast_to_size_internal(TracedRArray{T,0}((), get_mlir_data(arg), ()), rsize)
end

function broadcast_to_size(arg::AnyTracedRArray{T,0}, rsize) where {T}
    arg = materialize_traced_array(arg)
    return broadcast_to_size(TracedRNumber{T}((), get_mlir_data(arg)), rsize)
end

function broadcast_to_size(arg::AnyTracedRArray, rsize)
    arg = materialize_traced_array(arg)
    size(arg) == Tuple(rsize) && return arg
    return broadcast_to_size_internal(arg, rsize)
end

function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    rsize2 = (keep ? rsizev : 1 for (keep, rsizev) in zip(arg.keeps, rsize))
    x = broadcast_to_size(arg.x, rsize2)
    size(x) == rsize && return x
    return broadcast_to_size_internal(x, rsize)
end

@noinline function broadcast_to_size_internal(x::TracedRArray{T}, rsize) where {T}
    return Ops.broadcast_in_dim(x, collect(Int64, 1:ndims(x)), collect(Int64, rsize))
end

end
