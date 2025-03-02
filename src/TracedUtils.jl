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
using ReactantCore: MissingTracedValue, is_traced
using Functors: Functors

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
    @assert length(indices) == N "Expected $N indices, got $(length(indices))"
    indices = normalize_indices(x, indices...)
    if any(is_traced, indices)
        indices, integer_indices, result_size, _, flattened_size = traced_indices(
            indices...
        )
        linear_indices = mapreduce(+, enumerate(indices)) do (i, idx)
            bcasted_idxs = Ops.broadcast_in_dim(
                idx, ndims(idx) == 0 ? Int64[] : Int64[i], flattened_size
            )
            Base.stride(x, i) .* (bcasted_idxs .- 1)
        end
        linear_indices = linear_indices .+ 1
        parent_linear_indices_all = collect(LinearIndices(size(parent(x))))
        parent_linear_indices = promote_to(
            TracedRArray{Int64,ndims(parent_linear_indices_all)}, parent_linear_indices_all
        )[linear_indices]
        isempty(integer_indices) || (
            parent_linear_indices = materialize_traced_array(
                dropdims(parent_linear_indices; dims=integer_indices)
            )
        )
        parent_linear_indices = Ops.reshape(parent_linear_indices, result_size)
        return (parent_linear_indices,)
    else
        # Have this as a separate code-path since we can generate non-dynamic indexing
        cartesian_indices = CartesianIndex.(Iterators.product(indices...))
        linear_indices = LinearIndices(size(x))[cartesian_indices]
        parent_linear_indices = LinearIndices(size(parent(x)))[linear_indices]
        return (parent_linear_indices,)
    end
end

function set_mlir_data!(
    x::PermutedDimsArray{TracedRNumber{T},N,perm,iperm,TracedRArray{T,N}}, data
) where {T,N,perm,iperm}
    parent(x).mlir_data = permutedims(TracedRArray{T}(data), iperm).mlir_data
    return x
end

function set_mlir_data!(x::AnyTracedRArray{T}, data) where {T}
    ancestor_indices = get_ancestor_indices(x, axes(x)...)
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

mutable struct CompiledMlirFnResult{
    F,TR,Re,Rt,LA,LR,PA,CR,M<:Union{Nothing,Reactant.Sharding.Mesh},MA
}
    fnwrapped::Bool
    f::F
    traced_result::TR
    result::Re
    seen_args::OrderedIdDict
    ret::Rt
    linear_args::Vector{LA}
    in_tys::Vector{MLIR.IR.Type}
    linear_results::Vector{LR}
    num_partitions::Int
    num_replicas::Int
    is_sharded::Bool
    preserved_args::PA
    concrete_result::CR
    sharding_mesh::M
    mutated_args::MA
end

function make_mlir_fn(
    f,
    args,
    kwargs,
    name="main",
    concretein=true;
    toscalar=false,
    return_dialect=:func,
    args_in_result::Symbol=:all,
    construct_function_without_args::Bool=false,
    do_transpose=true,
    input_shardings=nothing, # This is not meant to be used by the user.
)
    if sizeof(typeof(f)) != 0 || f isa Base.BroadcastFunction
        mlir_fn_res = make_mlir_fn(
            Reactant.apply,
            (f, args...),
            kwargs,
            name,
            concretein;
            toscalar,
            return_dialect,
            do_transpose,
            args_in_result,
            input_shardings,
        )
        mlir_fn_res.fnwrapped = true
        return mlir_fn_res
    end

    num_partitions, num_replicas = 1, 1

    N = length(args)
    seen_args = OrderedIdDict()
    traced_args = Vector{Any}(undef, N)
    for i in 1:N
        @inbounds traced_args[i] = Reactant.make_tracer(
            seen_args,
            args[i],
            (:args, i),
            concretein ? Reactant.ConcreteToTraced : Reactant.TracedSetPath;
            toscalar,
        )
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
    end

    in_tys = if toscalar
        [
            MLIR.IR.TensorType((), MLIR.IR.Type(Reactant.unwrapped_eltype(arg))) for
            arg in linear_args
        ]
    elseif do_transpose
        [transpose_ty(Ops.mlir_type(arg)) for arg in linear_args]
    else
        [Ops.mlir_type(arg) for arg in linear_args]
    end

    sym_visibility = nothing
    if !concretein
        sym_visibility = MLIR.IR.Attribute("private")
    end

    ctx = MLIR.IR.context()
    mod = MLIR.IR.mmodule()

    # Insert meshes for the sharded arguments
    traced_args_to_shardings = OrderedIdDict()
    for (k, v) in seen_args
        if (k isa Reactant.ConcretePJRTArray || k isa Reactant.ConcretePJRTNumber)
            if Reactant.Sharding.is_sharded(k)
                Reactant.Ops.mesh(k.sharding.mesh)
                traced_args_to_shardings[v] = k.sharding
            elseif input_shardings !== nothing && haskey(input_shardings, k)
                Reactant.Ops.mesh(input_shardings[k].mesh)
                traced_args_to_shardings[v] = input_shardings[k]
            end
        end
    end

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

        if isempty(kwargs)
            Reactant.call_with_reactant(f, traced_args...)
        else
            Reactant.call_with_reactant(Core.kwcall, kwargs, f, traced_args...)
        end
    finally
        MLIR.IR.deactivate!(fnbody)
    end

    # check which arguments have been mutated
    mutated_args = Int[]
    if !construct_function_without_args
        for (i, arg) in enumerate(linear_args)
            if get_mlir_data(arg) != MLIR.IR.argument(fnbody, i)
                # mutation occured!
                push!(mutated_args, i)
            end
        end
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
        (args_in_result != :all && has_argidx(v)) && continue
        push!(linear_results, v)
    end
    if args_in_result == :mutated
        append!(linear_results, linear_args[mutated_args])
    end

    out_tys = if do_transpose
        [transpose_ty(Ops.mlir_type(arg)) for arg in linear_results]
    else
        [Ops.mlir_type(arg) for arg in linear_results]
    end

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
        args_in_result == :all && @assert length(vals) == length(linear_results)

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
            arg_attrs=MLIR.IR.attr(func, "arg_attrs"),
            res_attrs=MLIR.IR.attr(func, "res_attrs"),
            no_inline=MLIR.IR.attr(func, "no_inline"),
            sym_visibility,
        )
    end
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func2, 1), MLIR.IR.region(func, 1))

    mesh_cache = Reactant.Compiler.sdycache()
    is_sharded = !isempty(mesh_cache)

    if is_sharded
        unique_meshes = keys(mesh_cache)

        # TODO: support multiple meshes
        if length(unique_meshes) > 1
            error("Currently we support using a single mesh")
            sorted_devices = [sort(vec(m.device_ids)) for m in unique_meshes]
            @assert allequal(sorted_devices) "All meshes must have the same device ids"
        end
        sharding_mesh = first(unique_meshes)
        num_partitions = length(sharding_mesh)

        linear_arg_shardings = Vector{MLIR.IR.Attribute}(undef, length(linear_args))

        # Attach `sdy.sharding` attribute to the argument
        for (i, arg) in enumerate(linear_args)
            if haskey(traced_args_to_shardings, arg)
                sharding = traced_args_to_shardings[arg]
                (; sym_name, mesh_attr) = mesh_cache[sharding.mesh]
                linear_arg_shardings[i] = Reactant.Sharding.get_shardy_tensor_sharding_attribute(
                    sharding, ctx, sym_name, mesh_attr
                )
                MLIR.API.mlirFuncSetArgAttr(
                    func2, i - 1, "sdy.sharding", linear_arg_shardings[i]
                )
            end
        end

        # Ensure the sharding of the mutated arguments is propagated to the results
        result_not_replicated = falses(length(linear_results))
        for i in mutated_args
            arg = linear_args[i]
            if has_residx(arg) && haskey(traced_args_to_shardings, arg)
                residx = findfirst(Base.Fix1(===, arg), linear_results)
                @assert residx !== nothing
                result_not_replicated[residx] = true
                MLIR.API.mlirFuncSetResultAttr(
                    func2, residx - 1, "sdy.sharding", linear_arg_shardings[i]
                )
            end
        end
    else
        sharding_mesh = nothing
    end

    MLIR.API.mlirOperationDestroy(func.operation)
    func.operation = MLIR.API.MlirOperation(C_NULL)

    return CompiledMlirFnResult(
        false,
        func2,
        traced_result,
        result,
        seen_args,
        ret,
        linear_args,
        in_tys,
        linear_results,
        num_partitions,
        num_replicas,
        is_sharded,
        nothing,
        nothing,
        sharding_mesh,
        mutated_args,
    )
end

function __lookup_unique_name_in_module(mod, name)
    new_name = name
    tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
    for i in 0:10000
        new_name = i == 0 ? name : name * "_" * string(i)
        MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, new_name)) && return new_name
    end
    modstr = string(mod)
    return error("Mod\n$modstr\nCould not find unique name for $name")
end

function __take_region(compiled_fn)
    region = MLIR.IR.Region()
    MLIR.API.mlirRegionTakeBody(region, MLIR.API.mlirOperationGetRegion(compiled_fn, 0))
    return region
end

elem_apply(::Type{T}, x::TracedRArray{T}) where {T} = x

struct TypeCast{T<:Reactant.ReactantPrimitive} <: Function end

function (::TypeCast{T})(x::TracedRNumber{T2}) where {T,T2}
    return promote_to(TracedRNumber{T}, x)
end

function elem_apply(::Type{T}, x::TracedRArray) where {T<:Reactant.ReactantPrimitive}
    return elem_apply(TypeCast{T}(), x)
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

    set_mlir_data!(x, tostore)

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

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    if all(iszero ∘ ndims, args)
        scalar_args = map(args) do arg
            return promote_to(TracedRNumber{Reactant.unwrapped_eltype(arg)}, arg)
        end
        return f(scalar_args...)
    end

    mlir_fn_res = make_mlir_fn(
        f, args, (), string(f) * "_broadcast_scalar", false; toscalar=true
    )
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f
    (; result, seen_args, linear_args, linear_results) = mlir_fn_res

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

    out_tys2 = [
        MLIR.IR.TensorType(OutShape, MLIR.IR.Type(Reactant.unwrapped_eltype(arg))) for
        arg in linear_results
    ]

    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = get_argidx(a)
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
        if has_residx(a)
            path = get_residx(a)
            set!(result, path[2:end], MLIR.IR.result(res, residx))
            residx += 1
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                set!(f, path[3:end], MLIR.IR.result(res, residx))
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                set!(args[idx], path[3:end], MLIR.IR.result(res, residx))
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

function broadcast_to_size(arg::AbstractArray{<:TracedRNumber}, rsize)
    if Reactant.ancestor(arg) isa TracedRArray
        return broadcast_to_size(materialize_traced_array(arg), rsize)
    end
    return broadcast_to_size(reshape(Ops.vcat(arg...), size(arg)...), rsize)
end
broadcast_to_size(arg::AbstractArray, rsize) = broadcast_to_size(Ops.constant(arg), rsize)

broadcast_to_size(arg::AbstractRange, rsize) = broadcast_to_size(collect(arg), rsize)
function broadcast_to_size(arg::UnitRange, rsize)
    # For small inputs this will be automatically optimized away, and for large ranges
    # helps reduce the IR size
    x = Ops.add(
        Ops.iota(eltype(arg), [length(arg)]; iota_dimension=1),
        Ops.fill(first(arg), [length(arg)]),
    )
    return broadcast_to_size(x, rsize)
end
broadcast_to_size(arg::Base.OneTo, rsize) = broadcast_to_size(1:last(arg), rsize)

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

function normalize_indices(a::AbstractArray, indices...)
    return map(enumerate(indices)) do (i, idx)
        idx isa Colon && return collect(Int64, 1:size(a, i))
        idx isa CartesianIndex && return Tuple(idx)
        idx isa AbstractArray{Bool} && return findall(idx)
        return idx
    end
end

function traced_indices(indices...)
    integer_indices = Int64[]
    result_size = Int64[]
    preddim_result_size = Int64[]
    flattened_size = Int64[length(idx) for idx in indices]
    new_indices = map(enumerate(indices)) do (i, idx)
        if idx isa Number
            push!(preddim_result_size, 1)
            push!(integer_indices, i)
            idx isa TracedRNumber && return idx
            return promote_to(TracedRNumber{Int}, idx)
        end
        append!(preddim_result_size, [size(idx)...])
        append!(result_size, [size(idx)...])
        idx isa TracedRArray && return materialize_traced_array(vec(idx))
        return promote_to(TracedRArray{Int,1}, vec(idx))
    end
    return (
        new_indices,
        Tuple(integer_indices),
        result_size,
        preddim_result_size,
        flattened_size,
    )
end

end
