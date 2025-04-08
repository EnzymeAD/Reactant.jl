# Functions within this module and Ops do not get forcibly re-compiled to be within our interpreter.
# This means that replacements, for example, for autodiff/random/kernels/etc do not get applied here when
# within compilation. However, it means these functions are a _lot_ faster to compile.
module TracedUtils

using ..Reactant:
    Reactant,
    MLIR,
    RNumber,
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    MissingTracedValue,
    OrderedIdDict,
    ReactantPrimitive,
    Ops
using ReactantCore: ReactantCore
using ReactantCore: MissingTracedValue, is_traced, materialize_traced_array
using Functors: Functors

ReactantCore.materialize_traced_array(x::AbstractArray) = x

ReactantCore.materialize_traced_array(x::TracedRArray) = x

ReactantCore.materialize_traced_array(x::AnyTracedRArray) = x[axes(x)...]

function ReactantCore.materialize_traced_array(x::AbstractRange)
    return Reactant.aos_to_soa(collect(x))
end

function ReactantCore.materialize_traced_array(x::Base.OneTo)
    return Ops.iota(Reactant.unwrapped_eltype(x), [length(x)]; iota_dimension=1)
end

function ReactantCore.materialize_traced_array(x::UnitRange)
    return Ops.add(
        Ops.iota(Reactant.unwrapped_eltype(x), [length(x)]; iota_dimension=1),
        Ops.fill(first(x), [length(x)]),
    )
end

function ReactantCore.materialize_traced_array(x::SubArray)
    z = SubArray(materialize_traced_array(parent(x)), x.indices)
    return z[axes(z)...]
end

function ReactantCore.materialize_traced_array(x::Base.ReshapedArray)
    return Ops.reshape(materialize_traced_array(parent(x)), size(x)...)
end

function ReactantCore.materialize_traced_array(
    x::PermutedDimsArray{<:Any,<:Any,perm}
) where {perm}
    return permutedims(materialize_traced_array(parent(x)), perm)
end

function ReactantCore.materialize_traced_array(x::AbstractArray{TracedRNumber{T}}) where {T}
    return Reactant.aos_to_soa(x)
end

get_mlir_data(x::TracedRNumber) = x.mlir_data
set_mlir_data!(x::TracedRNumber, data) = (x.mlir_data=data; return x)
get_paths(x::TracedRNumber) = x.paths
set_paths!(x::TracedRNumber, paths) = (x.paths=paths; return x)

get_mlir_data(x::TracedRArray) = x.mlir_data
get_mlir_data(x::AnyTracedRArray) = get_mlir_data(materialize_traced_array(x))
get_paths(x::TracedRArray) = x.paths
set_paths!(x::TracedRArray, paths) = (x.paths=paths; return x)

get_paths(x::MissingTracedValue) = x.paths
set_paths!(x::MissingTracedValue, paths) = (x.paths=paths; return x)

function set_mlir_data!(x::TracedRArray, data)
    x.mlir_data = data
    return x
end

function set_mlir_data!(x::Base.ReshapedArray{TracedRNumber{T}}, data) where {T}
    set_mlir_data!(
        parent(x), get_mlir_data(Ops.reshape(TracedRArray{T}(data), size(parent(x))...))
    )
    return x
end

function get_ancestor_indices(
    x::Base.ReshapedArray{TracedRNumber{T},N}, indices...
) where {T,N}
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
    x::PermutedDimsArray{TracedRNumber{T},N,perm,iperm}, data
) where {T,N,perm,iperm}
    set_mlir_data!(parent(x), get_mlir_data(permutedims(TracedRArray{T}(data), iperm)))
    return x
end

function set_mlir_data!(x::AnyTracedRArray{T}, data) where {T}
    ancestor_indices = get_ancestor_indices(x, axes(x)...)
    setindex!(Reactant.ancestor(x), TracedRArray{T}(data), ancestor_indices...)
    return x
end

get_ancestor_indices(::TracedRArray, indices...) = indices
get_ancestor_indices(::Array{<:TracedRNumber}, indices...) = indices
function get_ancestor_indices(x::AnyTracedRArray, indices...)
    return get_ancestor_indices(parent(x), Base.reindex(parentindices(x), indices)...)
end

Base.@nospecializeinfer function batch_ty(
    width::Int, @nospecialize(mlirty::MLIR.IR.Type)
)::MLIR.IR.Type
    return MLIR.IR.TensorType(Int[width, size(mlirty)...], eltype(mlirty))
end

Base.@nospecializeinfer function transpose_ty(
    @nospecialize(mlirty::MLIR.IR.Type)
)::MLIR.IR.Type
    return MLIR.IR.TensorType(Int[reverse(size(mlirty))...], eltype(mlirty))
end

Base.@nospecializeinfer function transpose_val(
    @nospecialize(val::MLIR.IR.Value)
)::MLIR.IR.Value
    val_size = size(MLIR.IR.type(val))
    val_size == () && return val
    attr = MLIR.IR.DenseArrayAttribute(Int64[reverse(0:(length(val_size) - 1))...])
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val; permutation=attr), 1)
end

Base.@nospecializeinfer function unpad_val_op(
    @nospecialize(val::MLIR.IR.Value), padding, sz
)::MLIR.IR.Operation
    start_indices = zeros(Int64, length(padding))
    limit_indices = collect(Int64, sz) .- padding
    return MLIR.Dialects.stablehlo.slice(
        val;
        start_indices=MLIR.IR.DenseArrayAttribute(start_indices),
        limit_indices=MLIR.IR.DenseArrayAttribute(limit_indices),
        strides=MLIR.IR.DenseArrayAttribute(ones(Int64, length(padding))),
    )
end

mutable struct CompiledMlirFnResult{F,TR,Re,Rt,LA,LR,PA,CR,M,MA,RS,GD,DA}
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
    unique_meshes::M
    mutated_args::MA
    use_shardy_partitioner::Bool
    result_shardings::RS
    global_device_ids::GD # only populated if is_sharded
    donated_args_mask::DA
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
    input_shardings=nothing,  # This is not meant to be used by the user.
    output_shardings=nothing, # This is not meant to be used by the user.
    runtime=nothing,
    verify_arg_names=nothing,
    argprefix::Symbol=:args,
    resprefix::Symbol=:result,
    resargprefix::Symbol=:resargs,
    num_replicas=1,
    optimize_then_pad::Bool=true,
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
            args_in_result,
            construct_function_without_args,
            do_transpose,
            input_shardings,
            output_shardings,
            runtime,
            verify_arg_names,
            argprefix,
            resprefix,
            resargprefix,
            num_replicas,
        )
        mlir_fn_res.fnwrapped = true
        return mlir_fn_res
    end

    N = length(args)
    seen_args = OrderedIdDict()
    traced_args = Vector{Any}(undef, N)
    inmode = if concretein
        @assert !toscalar
        Reactant.ConcreteToTraced
    else
        Reactant.TracedSetPath
    end
    for i in 1:N
        @inbounds traced_args[i] = Reactant.make_tracer(
            seen_args, args[i], (argprefix, i), inmode; toscalar, runtime
        )
    end

    linear_args = Reactant.TracedType[]
    inv_map = IdDict()
    for (k, v) in seen_args
        v isa Reactant.TracedType || continue
        push!(linear_args, v)
        inv_map[v] = k
    end

    in_tys = Vector{MLIR.IR.Type}(undef, length(linear_args))
    for (i, arg) in enumerate(linear_args)
        elT = MLIR.IR.Type(Reactant.unwrapped_eltype(arg))
        if toscalar
            in_tys[i] = MLIR.IR.TensorType(Int[], elT)
        else
            sz = collect(Int, size(arg))
            if !optimize_then_pad
                carg = inv_map[arg]
                Reactant.has_padding(carg) && (sz .+= Reactant.get_padding(carg))
            end

            typ = MLIR.IR.TensorType(sz, elT)
            do_transpose && (typ = transpose_ty(typ))
            in_tys[i] = typ
        end
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
        if k isa Reactant.AbstractConcreteNumber || k isa Reactant.AbstractConcreteArray
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
            function_type=MLIR.IR.FunctionType(in_tys, Vector{MLIR.IR.Type}(undef, 0)),
            body=MLIR.IR.Region(),
        )
    end

    arglocs = MLIR.IR.Location[]
    for arg in linear_args
        path = get_idx(arg, argprefix)
        stridx = if verify_arg_names isa Nothing
            "arg" * string(path[2])
        else
            string(verify_arg_names.args[path[2]])
        end
        aval = args[path[2]]
        for (cidx, idx) in enumerate(path[3:end])
            if aval isa Array || aval isa Dict
                aval = getindex(aval, idx)
                stridx = stridx * "[" * string(idx) * "]"
            else
                fldname = if idx isa Integer
                    string(fieldname(Core.Typeof(aval), idx))
                else
                    string(idx)
                end
                stridx *= "." * fldname
                aval = getfield(aval, idx)
            end
        end
        push!(arglocs, MLIR.IR.Location(stridx * " (path=$path)", MLIR.IR.Location()))
    end
    fnbody = MLIR.IR.Block(in_tys, arglocs)
    push!(MLIR.IR.region(func, 1), fnbody)
    Ops.activate_constant_context!(fnbody)

    @assert MLIR.IR._has_block()

    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues
    MLIR.IR.activate!(fnbody)

    result = try
        for (i, arg) in enumerate(linear_args)
            raw_arg = MLIR.IR.argument(fnbody, i)
            row_maj_arg = do_transpose ? transpose_val(raw_arg) : raw_arg
            if !optimize_then_pad
                carg = inv_map[arg]
                if Reactant.has_padding(carg)
                    padding = Reactant.get_padding(carg)
                    sz = size(carg) .+ padding
                    if !do_transpose
                        padding = reverse(padding)
                        sz = reverse(sz)
                    end
                    row_maj_arg = MLIR.IR.result(unpad_val_op(row_maj_arg, padding, sz), 1)
                end
            end
            set_mlir_data!(arg, row_maj_arg)
        end

        if isempty(kwargs)
            Reactant.call_with_reactant(f, traced_args...)
        else
            Reactant.call_with_reactant(Core.kwcall, kwargs, f, traced_args...)
        end
    finally
        MLIR.IR.deactivate!(fnbody)
        Ops.deactivate_constant_context!(fnbody)
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

    outmode = if concretein
        @assert !toscalar
        Reactant.NoStopTracedTrack
    else
        Reactant.TracedTrack
    end

    MLIR.IR.activate!(fnbody)
    traced_result = try
        traced_result = Reactant.make_tracer(
            seen_results, result, (resprefix,), outmode; runtime
        )

        # marks buffers to be donated
        for i in 1:N
            Reactant.make_tracer(
                seen_results,
                traced_args[i],
                (resargprefix, i),
                Reactant.NoStopTracedTrack;
                runtime,
            )
        end
        traced_result
    finally
        MLIR.IR.deactivate!(fnbody)
    end

    linear_results = Reactant.TracedType[]
    for (k, v) in seen_results
        v isa Reactant.TracedType || continue
        if args_in_result != :all
            if has_idx(v, argprefix)
                if !(
                    (args_in_result == :result_and_mutated || args_in_result == :result) &&
                    has_idx(v, resprefix)
                )
                    continue
                end
            end
        end
        push!(linear_results, v)
    end

    if args_in_result == :mutated || args_in_result == :result_and_mutated
        append!(linear_results, linear_args[mutated_args])
    end
    if !isnothing(verify_arg_names) && typeof.(linear_args) != typeof.(linear_results)
        argis = []
        for arg in linear_args
            for path in arg.paths
                if length(path) == 0
                    continue
                end
                if path[1] != argprefix
                    continue
                end
                push!(argis, path[2:end])
            end
        end
        resis = []
        for arg in linear_results
            for path in arg.paths
                if length(path) == 0
                    continue
                end
                if path[1] != resargprefix
                    continue
                end
                push!(resis, path[2:end])
            end
        end

        # this can be more efficient

        err1 = []

        err2 = []
        for (errs, prev, post) in ((err1, resis, argis), (err2, argis, resis))
            conflicts = setdiff(prev, post)
            for conflict in conflicts
                stridx = string(verify_arg_names.args[conflict[1]])
                aval = args[conflict[1]]
                for (cidx, idx) in enumerate(Base.tail(conflict))
                    if aval isa Array || aval isa Dict
                        aval = Reactant.@allowscalar getindex(aval, idx)
                        stridx = stridx * "[" * string(idx) * "]"
                    else
                        fldname = if idx isa Integer
                            string(fieldname(Core.Typeof(aval), idx))
                        else
                            string(idx)
                        end
                        if cidx == 1
                            # Don't include the ref
                            if idx != 1
                                throw(
                                    AssertionError(
                                        "expected first path to be a ref lookup, found idx=$idx conflict=$conflict, cidx=$cidx",
                                    ),
                                )
                            end
                        else
                            stridx *= "." * fldname
                        end
                        aval = getfield(aval, idx)
                    end
                end
                push!(errs, stridx * " (path=$conflict, type=$(typeof(aval)))")
            end
        end

        arg_info = sort([(Base.pointer_from_objref(arg), arg.paths) for arg in linear_args])
        res_info = sort([
            (Base.pointer_from_objref(arg), arg.paths) for arg in linear_results
        ])

        arg_info_ni = [ai for ai in arg_info if !(ai in res_info)]
        res_info_ni = [ai for ai in res_info if !(ai in arg_info)]

        error("""Types do not match between function arguments and results.
        The following arguments should be traced but were not: $(join(err1, ", "))
        The following arguments should be returned but were not: $(join(err2, ", "))
        argprefix = $argprefix
        resprefix = $resprefix
        verify_arg_names = $verify_arg_names
        argtys = $(Core.Typeof.(args))
        Traced Arg Paths: \n$(join(arg_info, "\n"))\n
        Traced Res Paths: \n$(join(res_info, "\n"))\n
        Traced Arg NI Paths: \n$(join(arg_info_ni, "\n"))\n
        Traced Res NI Paths: \n$(join(res_info_ni, "\n"))\n
        traced_result : $(Core.Typeof.(traced_result))
        """)
    end

    out_tys = Vector{MLIR.IR.Type}(undef, length(linear_results))
    MLIR.IR.activate!(fnbody)
    ret = try
        vals = Vector{MLIR.IR.Value}(undef, length(linear_results))

        for (i, res) in enumerate(linear_results)
            if !optimize_then_pad && haskey(inv_map, res) && Reactant.has_padding(inv_map[res])
                carg = inv_map[res]
                padding = Reactant.get_padding(carg)
                sz = size(carg) .+ padding
                if !do_transpose
                    padding = reverse(padding)
                    sz = reverse(sz)
                end

                res = Ops.pad(
                    res,
                    promote_to(TracedRNumber{Reactant.unwrapped_eltype(res)}, 0);
                    high=collect(Int, padding),
                )
            end

            if res isa MissingTracedValue
                col_maj = get_mlir_data(broadcast_to_size(false, ()))
                out_ty = Ops.mlir_type(TracedRArray{Bool,0}, ())
            else
                col_maj = get_mlir_data(res)
                out_ty = Ops.mlir_type(res)

                if do_transpose
                    col_maj = transpose_val(col_maj)
                    out_ty = transpose_ty(out_ty)
                end
            end

            vals[i] = col_maj
            out_tys[i] = out_ty
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
    is_sharded =
        !isempty(mesh_cache) || (output_shardings !== nothing && !isempty(output_shardings))

    if is_sharded
        linear_arg_shardings = Vector{Tuple{MLIR.IR.Attribute,Symbol}}(
            undef, length(linear_args)
        )

        # If an argument is mutated but is not sharded (aka sharding is NoSharding), we
        # need to force a replicated sharding.
        for i in mutated_args
            arg = linear_args[i]
            if !haskey(traced_args_to_shardings, arg) && !isempty(mesh_cache)
                # Force a replicated sharding (it doesn't matter with mesh we use)
                traced_args_to_shardings[arg] = Reactant.Sharding.Replicated(
                    first(values(mesh_cache)).mesh
                )
            end
        end

        # Attach `sdy.sharding` attribute to the argument
        for (i, arg) in enumerate(linear_args)
            if haskey(traced_args_to_shardings, arg)
                sharding = traced_args_to_shardings[arg]
                (; sym_name, mesh_attr) = mesh_cache[(
                    sharding.mesh.logical_device_ids,
                    sharding.mesh.axis_names,
                    size(sharding.mesh),
                )]
                attr, dialect = Reactant.Sharding.get_tensor_sharding_attribute(
                    sharding, ctx, sym_name, mesh_attr, size(arg)
                )
                linear_arg_shardings[i] = (attr, dialect)
                if dialect == :sdy
                    MLIR.API.mlirFuncSetArgAttr(func2, i - 1, "sdy.sharding", attr)
                elseif dialect == :mhlo
                    MLIR.API.mlirFuncSetArgAttr(func2, i - 1, "mhlo.sharding", attr)
                else
                    error("Unsupported dialect for tensor sharding: $(dialect)")
                end
            end
        end

        # Ensure the sharding of the mutated arguments is propagated to the results
        for i in mutated_args
            arg = linear_args[i]

            if haskey(traced_args_to_shardings, arg) &&
                (has_idx(arg, resprefix) || has_idx(arg, resargprefix))
                idx = findfirst(Base.Fix1(===, arg), linear_results)
                @assert idx !== nothing
                attr, dialect = linear_arg_shardings[i]
                if dialect == :sdy
                    MLIR.API.mlirFuncSetResultAttr(func2, idx - 1, "sdy.sharding", attr)
                elseif dialect == :mhlo
                    MLIR.API.mlirFuncSetResultAttr(func2, idx - 1, "mhlo.sharding", attr)
                else
                    error("Unsupported dialect for tensor sharding: $(dialect)")
                end
            end
        end

        for (i, res) in enumerate(linear_results)
            if has_idx(res, argprefix) && haskey(traced_args_to_shardings, res)
                argidx = findfirst(Base.Fix1(===, res), linear_args)
                @assert argidx !== nothing
                attr, dialect = linear_arg_shardings[argidx]
                if dialect == :sdy
                    MLIR.API.mlirFuncSetResultAttr(func2, i - 1, "sdy.sharding", attr)
                elseif dialect == :mhlo
                    MLIR.API.mlirFuncSetResultAttr(func2, i - 1, "mhlo.sharding", attr)
                else
                    error("Unsupported dialect for tensor sharding: $(dialect)")
                end
            end
        end

        # XXX: Generalize the output shardings and expose it to the user
        # output_shardings is a Int -> Sharding mapping
        if output_shardings !== nothing
            for (i, arg) in enumerate(linear_results)
                if haskey(output_shardings, i)
                    sharding = output_shardings[i]
                    key = (
                        sharding.mesh.logical_device_ids,
                        sharding.mesh.axis_names,
                        size(sharding.mesh),
                    )
                    haskey(mesh_cache, key) || Reactant.Ops.mesh(sharding.mesh)
                    (; sym_name, mesh_attr) = mesh_cache[key]
                    attr, dialect = Reactant.Sharding.get_tensor_sharding_attribute(
                        sharding, ctx, sym_name, mesh_attr, size(arg)
                    )
                    if dialect == :sdy
                        MLIR.API.mlirFuncSetResultAttr(func2, i - 1, "sdy.sharding", attr)
                    elseif dialect == :mhlo
                        MLIR.API.mlirFuncSetResultAttr(func2, i - 1, "mhlo.sharding", attr)
                    else
                        error("Unsupported dialect for tensor sharding: $(dialect)")
                    end
                end
            end
        end

        unique_meshes = [m.mesh for m in values(mesh_cache)]
        sorted_devices = [m.device_ids for m in unique_meshes]
        @assert allequal(sorted_devices) "All meshes must have the same device ids"
        global_device_ids = first(sorted_devices)
        num_partitions = length(first(unique_meshes)) ÷ num_replicas
    else
        global_device_ids = ()
        unique_meshes = nothing
        num_partitions = 1
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
        unique_meshes,
        mutated_args,
        true,
        missing,
        global_device_ids,
        nothing, # populated later in `compile_mlir!`
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

function get_idx(x, prefix::Symbol)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == prefix
            return path
        end
    end
    throw(AssertionError("No path found for $x"))
end

function get_argidx(x, prefix::Symbol)
    path = get_idx(x, prefix)
    return path[2]::Int, path
end

function has_idx(x, prefix::Symbol)
    for path in get_paths(x)
        if length(path) == 0
            continue
        end
        if path[1] == prefix
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

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    if all(iszero ∘ ndims, args)
        scalar_args = map(args) do arg
            return promote_to(TracedRNumber{Reactant.unwrapped_eltype(arg)}, arg)
        end
        return f(scalar_args...)
    end

    argprefix::Symbol = gensym("broadcastarg")
    resprefix::Symbol = gensym("broadcastresult")
    resargprefix::Symbol = gensym("broadcastresarg")

    mlir_fn_res = make_mlir_fn(
        f,
        args,
        (),
        string(f) * "_broadcast_scalar",
        false;
        toscalar=true,
        argprefix,
        resprefix,
        resargprefix,
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

    out_tys2 = MLIR.IR.Type[
        MLIR.IR.TensorType(
            collect(Int, OutShape), MLIR.IR.Type(Reactant.unwrapped_eltype(arg))
        ) for arg in linear_results
    ]

    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = get_argidx(a, argprefix)
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
        resv = MLIR.IR.result(res, residx)
        residx += 1
        for path in a.paths
            if length(path) == 0
                continue
            end
            if path[1] == resprefix
                set!(result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if idx == 1 && fnwrap
                    set!(f, path[3:end], resv)
                else
                    if fnwrap
                        idx -= 1
                    end
                    set!(args[idx], path[3:end], resv)
                end
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

function broadcast_to_size(arg::AnyTracedRArray, rsize)
    if Reactant.isa_traced_soa(Reactant.ancestor(arg))
        return broadcast_to_size(materialize_traced_array(arg), rsize)
    end
    x = Reactant.aos_to_soa(arg)
    x === arg && return broadcast_to_size(materialize_traced_array(arg), rsize)
    return broadcast_to_size(x, rsize)
end

broadcast_to_size(arg::TracedRArray, rsize) = broadcast_to_size_internal(arg, rsize)

broadcast_to_size(arg::AbstractArray, rsize) = broadcast_to_size(Ops.constant(arg), rsize)

function broadcast_to_size(arg::AbstractRange{<:TracedRNumber}, rsize)
    return broadcast_to_size(collect(arg), rsize)
end
broadcast_to_size(arg::AbstractRange, rsize) = broadcast_to_size(collect(arg), rsize)

function broadcast_to_size(arg::UnitRange{<:TracedRNumber}, rsize)
    return @invoke broadcast_to_size(arg::UnitRange, rsize)
end
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

function broadcast_to_size(arg::AbstractIrrational, rsize)
    return broadcast_to_size(Base.convert(Float64, arg), rsize)
end

function broadcast_to_size(arg::ReactantPrimitive, rsize)
    return Ops.fill(arg, rsize)
end

function broadcast_to_size(arg::TracedRNumber{T}, rsize) where {T}
    length(rsize) == 0 && return arg
    return broadcast_to_size_internal(TracedRArray{T,0}((), get_mlir_data(arg), ()), rsize)
end

function broadcast_to_size(arg::AbstractArray{TracedRNumber{T},0}, rsize) where {T}
    arg = materialize_traced_array(arg)
    return broadcast_to_size(TracedRNumber{T}((), get_mlir_data(arg)), rsize)
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
