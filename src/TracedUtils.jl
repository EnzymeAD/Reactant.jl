# Functions within this module and Ops do not get forcibly re-compiled to be within our interpreter.
# This means that replacements, for example, for autodiff/random/kernels/etc do not get applied here when
# within compilation. However, it means these functions are a _lot_ faster to compile.
module TracedUtils

using ..Reactant:
    Reactant,
    MLIR,
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    MissingTracedValue,
    OrderedIdDict,
    Ops,
    promote_to, # keep this to avoid breaking external code
    broadcast_to_size # keep this to avoid breaking external code
using ..Ops: @opcall
using GPUArraysCore: @allowscalar
using ReactantCore: ReactantCore
using ReactantCore: MissingTracedValue, is_traced, materialize_traced_array

ReactantCore.materialize_traced_array(x::AbstractArray) = x

ReactantCore.materialize_traced_array(x::TracedRArray) = x

function ReactantCore.materialize_traced_array(x::AbstractRange)
    return Reactant.aos_to_soa(collect(x))
end

function ReactantCore.materialize_traced_array(r::LinRange)
    T = Reactant.unwrapped_eltype(r)
    idxs = @opcall iota(T, [length(r)]; iota_dimension=1)
    t = idxs ./ r.lendiv
    return T.((1 .- t) .* r.start .+ t .* r.stop)
end

function ReactantCore.materialize_traced_array(x::Base.OneTo)
    return @opcall iota(Reactant.unwrapped_eltype(x), [length(x)]; iota_dimension=1)
end

function ReactantCore.materialize_traced_array(x::UnitRange)
    return @opcall add(
        @opcall(iota(Reactant.unwrapped_eltype(x), [length(x)]; iota_dimension=1)),
        @opcall(fill(first(x), [length(x)])),
    )
end

function ReactantCore.materialize_traced_array(x::SubArray)
    return materialize_traced_array(parent(x))[Base.reindex(parentindices(x), axes(x))...]
end

function ReactantCore.materialize_traced_array(x::Base.ReshapedArray)
    if Base.prod(size(parent(x))) != Base.prod(size(x))
        throw(
            AssertionError(
                "Invalid reshape array, original size $(size(parent(x))) not compatible with new size $(size(x))",
            ),
        )
    end
    return @opcall reshape(materialize_traced_array(parent(x)), size(x)...)
end

function ReactantCore.materialize_traced_array(
    x::PermutedDimsArray{<:Any,<:Any,perm}
) where {perm}
    return permutedims(materialize_traced_array(parent(x)), perm)
end

function ReactantCore.materialize_traced_array(x::AbstractArray{TracedRNumber{T}}) where {T}
    as = Reactant.aos_to_soa(x)
    if as === x
        as = x[axes(x)...]
    end
    return ReactantCore.materialize_traced_array(as)
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

function set_mlir_data!(x::Base.ReshapedArray{TracedRNumber{T}}, data) where {T}
    set_mlir_data!(
        parent(x),
        get_mlir_data(@opcall(reshape(TracedRArray{T}(data), size(parent(x))...))),
    )
    return x
end

function get_ancestor_and_indices(
    x::Base.ReshapedArray{TracedRNumber{T},N}, indices::Vector{CartesianIndex{N}}
) where {T,N}
    linear_indices = LinearIndices(size(x))[indices]
    parent_linear_indices = LinearIndices(size(parent(x)))[linear_indices]
    return (parent(x), (parent_linear_indices,))
end

function get_ancestor_and_indices(
    x::Base.ReshapedArray{TracedRNumber{T},N}, indices...
) where {T,N}
    @assert length(indices) == N "Expected $N indices, got $(length(indices))"
    indices = Base.to_indices(x, indices)
    if any(is_traced, indices)
        indices, integer_indices, result_size, _, flattened_size = traced_indices(
            indices...
        )
        linear_indices = mapreduce(+, enumerate(indices)) do (i, idx)
            bcasted_idxs = @opcall broadcast_in_dim(
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
        parent_linear_indices = @opcall reshape(parent_linear_indices, result_size)
        return (parent(x), (parent_linear_indices,))
    else
        # Have this as a separate code-path since we can generate non-dynamic indexing
        cartesian_indices = CartesianIndex.(Iterators.product(indices...))
        linear_indices = LinearIndices(size(x))[cartesian_indices]
        parent_linear_indices = LinearIndices(size(parent(x)))[linear_indices]
        return (parent(x), (parent_linear_indices,))
    end
end

function set_mlir_data!(
    x::PermutedDimsArray{TracedRNumber{T},N,perm,iperm}, data
) where {T,N,perm,iperm}
    set_mlir_data!(parent(x), get_mlir_data(permutedims(TracedRArray{T}(data), iperm)))
    return x
end

function set_mlir_data!(x::AnyTracedRArray{T}, data) where {T}
    ancestor, ancestor_indices = get_ancestor_and_indices(x, axes(x)...)
    setindex!(Reactant.ancestor(x), TracedRArray{T}(data), ancestor_indices...)
    return x
end

get_ancestor_and_indices(a::TracedRArray, indices) = (a, indices)
get_ancestor_and_indices(a::TracedRArray, indices, args...) = (a, (indices, args...))

get_ancestor_and_indices(a::Array{<:TracedRNumber}, indices...) = (a, indices)
function get_ancestor_and_indices(a::Array{<:TracedRNumber}, indices, args...)
    return (a, (indices, args...))
end

function get_ancestor_and_indices(x::AnyTracedRArray, indices...)
    return get_ancestor_and_indices_inner(x, indices...) # redirect to avoid ambiguity
end
function get_ancestor_and_indices(x::AnyTracedRArray, indices, args...)
    return get_ancestor_and_indices_inner(x, indices, args...) # redirect to avoid ambiguity
end

function get_ancestor_and_indices_inner(
    x::AnyTracedRArray{T,N}, indices::Vararg{Any,N}
) where {T,N}
    return get_ancestor_and_indices(parent(x), Base.reindex(parentindices(x), indices)...)
end
function get_ancestor_and_indices_inner(x::AnyTracedRArray{T,1}, indices) where {T}
    return get_ancestor_and_indices(parent(x), Base.reindex(parentindices(x), indices))
end

function get_ancestor_and_indices_inner(
    x::AnyTracedRArray{T,N}, linear_indices::AbstractArray
) where {T,N}
    a, idxs = _get_ancestor_and_indices_linear(x, linear_indices)
    return a, (idxs isa Tuple ? idxs : (idxs,))
end
function get_ancestor_and_indices_inner(
    x::AnyTracedRArray{T,1}, linear_indices::AbstractArray
) where {T}
    a, idxs = _get_ancestor_and_indices_linear(x, linear_indices)
    return a, (idxs isa Tuple ? idxs : (idxs,))
end

function _get_ancestor_and_indices_linear(x::AnyTracedRArray, indices::AbstractArray)
    indices = CartesianIndices(x)[indices]
    pidxs = parentindices(x)
    parent_indices = map(indices) do idx
        CartesianIndex(Base.reindex(pidxs, (idx.I...,)))
    end
    return get_ancestor_and_indices(parent(x), parent_indices)
end

Base.@nospecializeinfer function batch_ty(
    width::Int, @nospecialize(mlirty::MLIR.IR.Type)
)::MLIR.IR.Type
    width == 1 && return mlirty
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
    skipped_args::Vector{LA}
    in_tys::Vector{MLIR.IR.Type}
    linear_results::Vector{LR}
    skipped_results::Vector{LR}
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
    is_pure::Bool
end

function is_pure(func)
    attr = MLIR.IR.attr(func, "enzymexla.memory_effects")
    # conservatively assume is not pure
    if attr isa Nothing
        return false
    end
    any(at -> String(at) == "write", attr) && return false
    return true
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
    within_autodiff=false,
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

    (; N, traced_args, linear_args, inv_map, in_tys, sym_visibility, mod, traced_args_to_shardings, func, fnbody, seen_args, skipped_args) = prepare_mlir_fn_args(
        args,
        name,
        concretein,
        toscalar,
        argprefix,
        runtime,
        optimize_then_pad,
        do_transpose,
        input_shardings,
        verify_arg_names,
    )

    Ops.activate_constant_context!(fnbody)
    @assert MLIR.IR._has_block()

    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues
    MLIR.IR.activate!(fnbody)

    result = try
        process_linear_args!(linear_args, fnbody, do_transpose, optimize_then_pad, inv_map)

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

    (func2, traced_result, ret, linear_args, in_tys, linear_results, skipped_results, num_partitions, is_sharded, unique_meshes, mutated_args, global_device_ids) = finalize_mlir_fn(
        result,
        traced_args,
        linear_args,
        skipped_args,
        seen_args,
        fnbody,
        func,
        mod,
        name,
        in_tys,
        do_transpose,
        optimize_then_pad,
        inv_map,
        args_in_result,
        resprefix,
        argprefix,
        resargprefix,
        verify_arg_names,
        return_dialect,
        traced_args_to_shardings,
        output_shardings,
        sym_visibility,
        num_replicas,
        runtime,
        construct_function_without_args,
        args,
        N,
        concretein,
        toscalar,
    )

    return CompiledMlirFnResult(
        false,
        func2,
        traced_result,
        result,
        seen_args,
        ret,
        linear_args,
        skipped_args,
        in_tys,
        linear_results,
        skipped_results,
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
        is_pure(func2),
    )
end

function prepare_mlir_fn_args(
    args,
    name,
    concretein,
    toscalar,
    argprefix,
    runtime,
    optimize_then_pad,
    do_transpose,
    input_shardings,
    verify_arg_names,
)
    N = length(args)
    traced_args = Vector{Any}(undef, N)
    inmode = if concretein
        @assert !toscalar
        Reactant.ConcreteToTraced
    else
        Reactant.TracedSetPath
    end
    fnbody = MLIR.IR.Block(MLIR.IR.Type[], MLIR.IR.Location[])
    MLIR.IR.activate!(fnbody)
    Ops.activate_constant_context!(fnbody)
    seen_args0 = OrderedIdDict()
    try
        for i in 1:N
            @inbounds traced_args[i] = Reactant.make_tracer(
                seen_args0, args[i], (argprefix, i), inmode; toscalar, runtime
            )
        end
    finally
        MLIR.IR.deactivate!(fnbody)
        Ops.deactivate_constant_context!(fnbody)
    end

    seen_args = OrderedIdDict()
    linear_args = Reactant.TracedType[]
    skipped_args = Reactant.TracedType[]
    inv_map = IdDict()
    for (k, v) in seen_args0
        v isa Reactant.TracedType || continue
        arg = get_mlir_data(v)
        if (arg isa MLIR.IR.Value) &&
            MLIR.IR.is_op_res(arg) &&
            MLIR.IR.block(MLIR.IR.op_owner(arg)) == fnbody
            push!(skipped_args, v)
            continue
        end
        seen_args[k] = v
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

    mod = MLIR.IR.mmodule()

    # Insert meshes for the sharded arguments
    traced_args_to_shardings = OrderedIdDict()
    for (k, v) in seen_args
        if k isa Reactant.AbstractConcreteNumber || k isa Reactant.AbstractConcreteArray
            if Reactant.Sharding.is_sharded(k)
                @opcall mesh(k.sharding.mesh)
                traced_args_to_shardings[v] = k.sharding
            elseif input_shardings !== nothing && haskey(input_shardings, k)
                @opcall mesh(input_shardings[k].mesh)
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

    for (i, arg) in enumerate(linear_args)
        path = get_idx(arg, argprefix)
        stridx = if verify_arg_names isa Nothing
            "arg" * string(path[2])
        else
            string(verify_arg_names[path[2]])
        end
        aval = args[path[2]]
        for idx in path[3:end]
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
                aval = Reactant.Compiler.traced_getfield(aval, idx)
            end
        end
        MLIR.IR.push_argument!(
            fnbody,
            in_tys[i];
            location=MLIR.IR.Location(stridx * " (path=$path)", MLIR.IR.Location()),
        )
    end
    push!(MLIR.IR.region(func, 1), fnbody)

    return (;
        N,
        traced_args,
        linear_args,
        inv_map,
        in_tys,
        sym_visibility,
        mod,
        traced_args_to_shardings,
        func,
        fnbody,
        seen_args,
        skipped_args,
    )
end

function process_linear_args!(linear_args, fnbody, do_transpose, optimize_then_pad, inv_map)
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
end

function finalize_mlir_fn(
    result,
    traced_args,
    linear_args,
    skipped_args,
    seen_args,
    fnbody,
    func,
    mod,
    name,
    in_tys,
    do_transpose,
    optimize_then_pad,
    inv_map,
    args_in_result,
    resprefix,
    argprefix,
    resargprefix,
    verify_arg_names,
    return_dialect,
    traced_args_to_shardings,
    output_shardings,
    sym_visibility,
    num_replicas,
    runtime,
    construct_function_without_args,
    args,
    N,
    concretein,
    toscalar,
)
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

    outmode = if concretein
        @assert !toscalar
        Reactant.NoStopTracedTrack
    else
        Reactant.TracedTrack
    end

    seen_results = OrderedIdDict()
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
    skipped_results = Reactant.TracedType[]
    for (k, v) in seen_results
        v isa Reactant.TracedType || continue
        if Reactant.looped_any(Base.Fix1(===, k), skipped_args)
            push!(skipped_results, v)

            _, argpath = get_argidx(v, argprefix)

            @assert has_idx(v, argprefix)

            newpaths = Tuple[]
            for path in v.paths
                if length(path) == 0
                    continue
                end
                if path[1] == argprefix
                    continue
                end
                if path[1] == resargprefix
                    original_arg = args[path[2]]
                    for p in path[3:end]
                        original_arg = Reactant.Compiler.traced_getfield(original_arg, p)
                    end
                    if !(
                        original_arg isa Union{
                            Reactant.ConcreteRNumber,
                            Reactant.ConcreteRArray,
                            Reactant.TracedType,
                        }
                    )
                        continue
                    end
                    push!(newpaths, path)
                end
                if path[1] == resprefix
                    push!(newpaths, path)
                end
            end

            if length(newpaths) != 0
                push!(linear_results, Reactant.repath(v, (newpaths...,)))
            end

            continue
        end
        if args_in_result != :all
            if has_idx(v, argprefix)
                if !(args_in_result == :result && has_idx(v, resprefix))
                    continue
                end
            end
        end
        push!(linear_results, v)
    end

    if args_in_result == :mutated
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
                stridx = string(verify_arg_names[conflict[1]])
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

                res = @opcall pad(
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

    mem = MLIR.IR.attr(func, "enzymexla.memory_effects")
    if !(mem isa Nothing)
        MLIR.IR.attr!(func2, "enzymexla.memory_effects", mem)
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

        ctx = MLIR.IR.context()
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
                    haskey(mesh_cache, key) || @opcall(mesh(sharding.mesh))
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

    return (
        func2,
        traced_result,
        ret,
        linear_args,
        in_tys,
        linear_results,
        skipped_results,
        num_partitions,
        is_sharded,
        unique_meshes,
        mutated_args,
        global_device_ids,
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

function __elem_apply_loop_condition(idx_ref, fn_ref::F, res_ref, args_ref, L_ref) where {F}
    return idx_ref[] < L_ref[]
end

function __elem_apply_loop_body(idx_ref, fn_ref::F, res_ref, args_ref, L_ref) where {F}
    args = args_ref[]
    fn = fn_ref[]
    res = res_ref[]
    idx = idx_ref[] + 1

    scalar_args = [@allowscalar(arg[idx]) for arg in args]
    @allowscalar res[idx] = fn(scalar_args...)

    idx_ref[] = idx
    res_ref[] = res
    return nothing
end

function elem_apply_via_while_loop(f, args::Vararg{Any,Nargs}) where {Nargs}
    @assert allequal(size.(args)) "All args must have the same size"
    L = length(first(args))
    # flattening the tensors makes the auto-batching pass work nicer
    flat_args = [ReactantCore.materialize_traced_array(vec(arg)) for arg in args]

    # This wont be a mutating function so we can safely execute it once
    res_tmp = @allowscalar(f([@allowscalar(arg[1]) for arg in flat_args]...))
    result = similar(first(flat_args), Reactant.unwrapped_eltype(res_tmp), L)

    ind_var = Ref(0)
    f_ref = Ref(f)
    result_ref = Ref(result)
    args_ref = Ref(flat_args)
    limit_ref = Ref(L)

    ReactantCore.traced_while(
        __elem_apply_loop_condition,
        __elem_apply_loop_body,
        (ind_var, f_ref, result_ref, args_ref, limit_ref),
    )

    return ReactantCore.materialize_traced_array(reshape(result, size(first(args))))
end

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    if all(iszero ∘ ndims, args)
        scalar_args = map(args) do arg
            return promote_to(TracedRNumber{Reactant.unwrapped_eltype(arg)}, arg)
        end
        return Reactant.call_with_reactant(f, scalar_args...)
    end

    # we can expand the scope of this later to support cases where the output
    # doesn't align with `Ops.batch`. For now we just handle cases that would
    # obviously fail with scalarizing the inputs.
    if Reactant.use_overlayed_version(f)
        return elem_apply_via_while_loop(f, args...)
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

    keys_seen = Reactant.TracedType[k for k in keys(seen_args) if k isa Reactant.TracedType]
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

_isone(x) = isone(x)
_isone(::CartesianIndex) = false

__contiguous_indices(::Base.LogicalIndex) = false
__contiguous_indices(x) = all(_isone, diff(x))

function create_index_mesh(idxs::AbstractVector...)
    lens = map(length, idxs)
    inner_repeats = cumprod(lens) .÷ lens
    outer_repeats = reverse(cumprod(reverse(lens)) .÷ reverse(lens))
    return [
        repeat(idx; inner, outer) for
        (idx, inner, outer) in zip(idxs, inner_repeats, outer_repeats)
    ]
end

function indices_to_gather_dims(indices...)
    non_contiguous_indices = TracedRArray{Int,1}[]
    contiguous_indices = Tuple{Int,Int}[]
    non_contiguous_indices_idxs = Int[]
    contiguous_indices_idxs = Int[]
    ddims = Int[]
    result_shape = Int64[]
    for (i, index) in enumerate(indices)
        if index isa Number
            push!(ddims, i)
            if index isa TracedRNumber
                push!(non_contiguous_indices_idxs, i)
                push!(non_contiguous_indices, broadcast_to_size(index, (1,)))
            else
                push!(contiguous_indices_idxs, i)
                push!(contiguous_indices, (index, 1))
            end
        else
            append!(result_shape, [size(index)...])
            if !(index isa TracedRArray)
                if __contiguous_indices(vec(index))
                    push!(contiguous_indices_idxs, i)
                    push!(contiguous_indices, (first(index), length(index)))
                    continue
                end
                index = promote_to(TracedRArray{Int,ndims(index)}, index)
            end
            push!(non_contiguous_indices_idxs, i)
            push!(non_contiguous_indices, materialize_traced_array(vec(index)))
        end
    end

    expanded_non_contiguous_indices = create_index_mesh(non_contiguous_indices...)
    L = length(first(expanded_non_contiguous_indices))
    new_indices = TracedRArray{Int,1}[]
    slice_sizes = ones(Int, length(indices))
    start_index_map = Int64[]

    for i in 1:length(indices)
        cont_idx = findfirst(==(i), contiguous_indices_idxs)
        if cont_idx !== nothing
            if !isone(contiguous_indices[cont_idx][1])
                push!(new_indices, broadcast_to_size(contiguous_indices[cont_idx][1], (L,)))
                push!(start_index_map, i)
            end
            slice_sizes[i] = contiguous_indices[cont_idx][2]
            continue
        end

        non_cont_idx = findfirst(==(i), non_contiguous_indices_idxs)
        @assert non_cont_idx !== nothing
        push!(new_indices, expanded_non_contiguous_indices[non_cont_idx])
        push!(start_index_map, i)
    end

    collapsed_slice_dims = vcat(non_contiguous_indices_idxs, ddims)
    sort!(collapsed_slice_dims)
    unique!(collapsed_slice_dims)
    start_indices = hcat(new_indices...)
    offset_dims = collect(Int64, 2:(length(indices) - length(collapsed_slice_dims) + 1))

    gather_reshape_shape = Int64[]
    perm = Int64[]
    for i in non_contiguous_indices_idxs
        push!(gather_reshape_shape, length(indices[i]))
        push!(perm, i)
    end
    for i in contiguous_indices_idxs
        push!(gather_reshape_shape, length(indices[i]))
        push!(perm, i)
    end

    return (;
        start_indices,
        slice_sizes,
        index_vector_dim=ndims(start_indices),
        start_index_map,
        collapsed_slice_dims,
        offset_dims,
        result_shape,
        permutation=invperm(perm),
        gather_reshape_shape,
    )
end

end
