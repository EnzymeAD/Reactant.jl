# Functions within this module and Ops do not get forcibly re-compiled to be within our interpreter.
# This means that replacements, for example, for autodiff/random/kernels/etc do not get applied here when
# within compilation. However, it means these functions are a _lot_ faster to compile.
module TracedUtils

using LinearAlgebra: LinearAlgebra
using Adapt: Adapt
using ..Reactant:
    RArray,
    RNumber,
    TracedRArray,
    TracedRNumber,
    WrappedTracedRArray,
    AnyTracedRArray,
    MissingTracedValue,
    OrderedIdDict
import ..Reactant
import ..Reactant.MLIR
import ..ReactantPrimitive
import ..Ops

materialize_traced_array(x::TracedRArray) = x
materialize_traced_array(x::WrappedTracedRArray) = x[axes(x)...]
function materialize_traced_array(
    x::Adapt.WrappedReshapedArray{T,N,<:TracedRArray}
) where {T,N}
    return Ops.reshape(materialize_traced_array(parent(x)), size(x)...)
end
function materialize_traced_array(
    x::LinearAlgebra.Transpose{T,TracedRArray{T,N}}
) where {T,N}
    px = parent(x)
    A = ndims(px) == 1 ? reshape(px, :, 1) : px
    return permutedims(A, (2, 1))
end
function materialize_traced_array(x::LinearAlgebra.Adjoint{T,TracedRArray{T,N}}) where {T,N}
    return conj(materialize_traced_array(transpose(parent(x))))
end
function materialize_traced_array(
    x::PermutedDimsArray{T,N,perm,iperm,<:TracedRArray{T,N}}
) where {T,N,perm,iperm}
    return permutedims(parent(x), perm)
end
function materialize_traced_array(x::LinearAlgebra.Diagonal{T,TracedRArray{T,1}}) where {T}
    return LinearAlgebra.diagm(parent(x))
end

get_mlir_data(x::TracedRNumber) = x.mlir_data
set_mlir_data!(x::TracedRNumber, data) = (x.mlir_data = data; return x)

get_mlir_data(x::TracedRArray) = x.mlir_data
get_mlir_data(x::AnyTracedRArray) = get_mlir_data(materialize_traced_array(x))

function set_mlir_data!(x::TracedRArray, data)
    x.mlir_data = data
    return x
end
function set_mlir_data!(x::Adapt.WrappedReshapedArray{T,N,<:TracedRArray}, data) where {T,N}
    res_mlir_data = Ops.reshape(TracedRArray(data), size(parent(x))...).mlir_data
    set_mlir_data!(parent(x), res_mlir_data)
    return x
end
function set_mlir_data!(x::LinearAlgebra.Transpose{T,TracedRArray{T,N}}, data) where {T,N}
    tdata = TracedRArray(data)
    px = parent(x)
    px.mlir_data = (
        if ndims(px) == 1
            Ops.reshape(tdata, length(tdata))
        else
            Ops.transpose(tdata, [2, 1])
        end
    ).mlir_data
    return x
end
function set_mlir_data!(x::LinearAlgebra.Adjoint{T,TracedRArray{T,N}}, data) where {T,N}
    tdata = TracedRArray(data)
    px = parent(x)
    transposed_data =
        ndims(px) == 1 ? Ops.reshape(tdata, length(tdata)) : Ops.transpose(tdata, [2, 1])
    px.mlir_data = (T <: Real ? transposed_data : Ops.conj(transposed_data)).mlir_data
    return x
end
function set_mlir_data!(
    x::PermutedDimsArray{T,N,perm,iperm,TracedRArray{T,N}}, data
) where {T,N,perm,iperm}
    parent(x).mlir_data = permutedims(TracedRArray(data), iperm).mlir_data
    return x
end
function set_mlir_data!(x::LinearAlgebra.Diagonal{T,TracedRArray{T,1}}, data) where {T}
    parent(x).mlir_data = LinearAlgebra.diag(TracedRArray(data)).mlir_data
    return x
end
function set_mlir_data!(x::AnyTracedRArray, data)
    setindex!(x, TracedRArray(data), axes(x)...)
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
    attr = MLIR.IR.DenseArrayAttribute(
        Int64[reverse(0:(length(size(MLIR.IR.type(val))) - 1))...]
    )
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val; permutation=attr), 1)
end

function make_mlir_fn(
    f,
    args,
    kwargs,
    name="main",
    concretein=true;
    toscalar=false,
    return_dialect=:func,
    no_args_in_result::Bool=false,
    construct_function_without_args::Bool=false,
    do_transpose=true,
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
                toscalar,
                return_dialect,
                no_args_in_result,
                construct_function_without_args,
                do_transpose,
            )[2:end]...,
        )
    end

    N = length(args)
    seen_args = OrderedIdDict()
    traced_args = ntuple(N) do i
        return Reactant.make_tracer(
            seen_args,
            args[i],
            (:args, i),
            concretein ? Reactant.ConcreteToTraced : Reactant.TracedSetPath;
            toscalar,
            track_numbers=construct_function_without_args ? (Number,) : (),
        )
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
    end

    in_tys = if toscalar
        [MLIR.IR.TensorType((), MLIR.IR.Type(eltype(arg))) for arg in linear_args]
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

    if construct_function_without_args
        fnbody = MLIR.IR.Block()
    else
        fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
    end
    push!(MLIR.IR.region(func, 1), fnbody)

    @assert MLIR.IR._has_block()

    result = MLIR.IR.block!(fnbody) do
        for (i, arg) in enumerate(linear_args)
            if construct_function_without_args
                arg.mlir_data = args[i].mlir_data
            else
                raw_arg = MLIR.IR.argument(fnbody, i)
                row_maj_arg = do_transpose ? transpose_val(raw_arg) : raw_arg
                arg.mlir_data = row_maj_arg
            end
        end

        # TODO fix it for kwargs
        #if concretein
        Reactant.call_with_reactant(f, traced_args...)
        #else
        #    f(traced_args...)
        #end
    end

    seen_results = OrderedIdDict()

    traced_result = Reactant.make_tracer(
        seen_results,
        result,
        (:result,),
        concretein ? Reactant.TracedTrack : Reactant.TracedSetPath;
        track_numbers=construct_function_without_args ? (Number,) : (),
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
        (no_args_in_result && length(v.paths) > 0 && v.paths[1][1] == :args) && continue
        push!(linear_results, v)
    end

    out_tys = [transpose_ty(Ops.mlir_type(arg)) for arg in linear_results]

    ret = MLIR.IR.block!(fnbody) do
        vals = MLIR.IR.Value[]
        for res in linear_results
            col_maj = if res isa MissingTracedValue
                broadcast_to_size(false, ()).mlir_data
            elseif construct_function_without_args || !do_transpose
                res.mlir_data
            elseif do_transpose
                transpose_val(res.mlir_data)
            end
            push!(vals, col_maj)
        end
        !no_args_in_result && @assert length(vals) == length(linear_results)

        dialect = getfield(MLIR.Dialects, return_dialect)
        return dialect.return_(vals)
    end

    name2 = name

    tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
    for i in 0:10000
        name2 = if i == 0
            name
        else
            name * string(i)
        end
        if MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, name2))
            break
        end
    end

    func2 = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=name2,
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

elem_apply(::Type{T}, x::TracedRArray{T}) where {T<:ReactantPrimitive} = x

struct TypeCast{T<:ReactantPrimitive} <: Function end

function (::TypeCast{T})(x::TracedRNumber{T2}) where {T,T2}
    return TracedUtils.promote_to(TracedRNumber{T}, x)
end

function elem_apply(
    ::Type{T}, x::TracedRArray{T2}
) where {T<:ReactantPrimitive,T2<:ReactantPrimitive}
    # Special Path to prevent going down a despecialized path
    return elem_apply(TypeCast{T}(), x)
end

function promote_to end

function get_attribute_by_name(operation, name)
    return MLIR.IR.Attribute(MLIR.API.mlirOperationGetAttributeByName(operation, name))
end

function push_val!(ad_inputs, x, path)
    for p in path
        x = traced_getfield(x, p)
    end
    x = x.mlir_data
    return push!(ad_inputs, x)
end

function get_argidx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :args
            return path[2]::Int, path
        end
    end
    throw(AssertionError("No path found for $x"))
end

function set!(x, path, tostore; emptypath=false)
    for p in path
        x = traced_getfield(x, p)
    end

    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function get_residx(x)
    for path in x.paths
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
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :result
            return true
        end
    end
    return false
end

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    if all(iszero ∘ ndims, args)
        scalar_args = map(args) do arg
            return promote_to(TracedRNumber{eltype(arg)}, arg)
        end
        return f(scalar_args...)
    end

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        f, args, (), string(f) * "_broadcast_scalar", false; toscalar=true
    )

    invmap = IdDict()
    for (k, v) in seen_args
        invmap[v] = k
    end

    keys_seen = [k for k in keys(seen_args) if k isa Reactant.TracedType]
    input_shapes = size.(keys_seen)
    # by the time we reach here all args must have same size
    @assert allequal(input_shapes) "input shapes are $(input_shapes)"
    OutShape = isempty(seen_args) ? nothing : first(input_shapes)
    @assert !isnothing(OutShape)

    in_tys2 = [Ops.mlir_type(invmap[arg]) for arg in linear_args]

    out_tys2 = [
        MLIR.IR.TensorType(OutShape, MLIR.IR.Type(eltype(arg))) for arg in linear_results
    ]

    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = TracedUtils.get_argidx(a)
        if idx == 1 && fnwrap
            push_val!(batch_inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    res = MLIR.Dialects.enzyme.batch(
        batch_inputs;
        outputs=out_tys2,
        fn=fname,
        batch_shape=MLIR.IR.DenseArrayAttribute([Int64(i) for i in OutShape]),
    )

    residx = 1

    for a in linear_results
        if TracedUtils.has_residx(a)
            path = TracedUtils.get_residx(a)
            TracedUtils.set!(result, path[2:end], MLIR.IR.result(res, residx))
            residx += 1
        else
            idx, path = TracedUtils.get_argidx(a)
            if idx == 1 && fnwrap
                TracedUtils.set!(f, path[3:end], MLIR.IR.result(res, residx))
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                TracedUtils.set!(args[idx], path[3:end], MLIR.IR.result(res, residx))
                residx += 1
            end
        end
    end

    seen_results = OrderedIdDict()
    traced2_result = Reactant.make_tracer(
        seen_results, result, (), Reactant.TracedSetPath; tobatch=OutShape
    )

    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return traced2_result
end

new_traced_value(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), nothing, size(A))
new_traced_value(::TracedRNumber{T}) where {T} = TracedRNumber{T}((), nothing)

broadcast_to_size(arg::AbstractArray, rsize) = broadcast_to_size(Ops.constant(arg), rsize)

function broadcast_to_size(arg::Base.RefValue, rsize)
    # XXX: don't we want to expand here to rsize?
    return arg
end

broadcast_to_size(arg::Number, rsize) = Ops.constant(Base.fill(arg, Tuple(rsize)))

function broadcast_to_size(arg::TracedRNumber, rsize)
    length(rsize) == 0 && return arg
    return broadcast_to_size_internal(
        TracedRArray{eltype(arg),0}((), arg.mlir_data, ()), rsize
    )
end

function broadcast_to_size(arg::AnyTracedRArray{T,0}, rsize) where {T}
    arg = materialize_traced_array(arg)
    return broadcast_to_size(TracedRNumber{T}((), arg.mlir_data), rsize)
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

@noinline function broadcast_to_size_internal(x::TracedRArray, rsize)
    dims = collect(Int64, 0:(length(size(x)) - 1))

    if length(size(MLIR.IR.type(x.mlir_data))) != length(dims)
        @show x
        @show arg
        @show rsize
        @show rsize2
        @show dims
    end
    @assert length(size(MLIR.IR.type(x.mlir_data))) == length(dims)
    mlirty = MLIR.IR.type(x.mlir_data)

    return TracedRArray{eltype(x),Int(length(rsize))}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.broadcast_in_dim(
                x.mlir_data;
                result_0=MLIR.IR.TensorType([t for t in rsize], eltype(mlirty)),
                broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims),
            ),
            1,
        ),
        collect(rsize),
    )
end

end