# This file defines the Julia code generation logic that is ultimately the body of a `Thunk`.

using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    ConcretePJRTArray,
    ConcretePJRTNumber,
    ConcreteIFRTArray,
    ConcreteIFRTNumber,
    AbstractConcreteArray,
    AbstractConcreteNumber,
    ancestor

const DEBUG_DISABLE_RESHARDING = Ref(false)
const DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR = Ref(false)
const DEBUG_BUFFER_POINTERS_STORE_DICT = Base.IdDict()

@inline function traced_getfield(@nospecialize(obj::Dict), field)
    return Base.getindex(obj, field)
end

@inline function traced_getfield(@nospecialize(obj), field)
    return Base.getfield(obj, field)
end

@inline function traced_getfield(
    @nospecialize(
        obj::AbstractArray{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}}
    ),
    field,
)
    return Base.getfield(obj, field)
end

@inline function traced_getfield(
    @nospecialize(obj::Array{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}}),
    field,
)
    return Base.getindex(obj, field)
end

@inline function traced_getfield(
    @nospecialize(
        obj::Union{AbstractConcreteArray,AbstractConcreteNumber}
    ),
    field,
)
    return Base.getproperty(obj, field)
end

@inline function traced_getfield(@nospecialize(obj::AbstractArray{T}), field) where {T}
    (isbitstype(T) || ancestor(obj) isa RArray || obj isa AbstractRange) &&
        return Base.getfield(obj, field)
    return Base.getindex(obj, field)
end

@inline function traced_setfield!(
    @nospecialize(obj::AbstractConcreteNumber), field, val, path
)
    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[] && field == :data
        if val ∈ keys(DEBUG_BUFFER_POINTERS_STORE_DICT)
            if obj !== DEBUG_BUFFER_POINTERS_STORE_DICT[val]
                error("Aliased buffer cannot be assigned to multiple Concrete Structs. \
                       Path: $path.")
            end
        else
            DEBUG_BUFFER_POINTERS_STORE_DICT[val] = obj
        end
    end

    return Base.setproperty!(obj, field, val)
end

@inline traced_setfield!(@nospecialize(obj), field, val, _path) =
    Base.setfield!(obj, field, val)

@inline function traced_setfield!(
    @nospecialize(obj::AbstractArray{T}), field, val, path
) where {T}
    ancestor_obj = ancestor(obj)
    (isbitstype(T) || ancestor_obj isa RArray) &&
        return setfield_carray!(obj, field, val, path)
    return Base.setindex!(obj, val, field)
end

@inline function traced_setfield!(
    @nospecialize(obj::AbstractArray{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber}}),
    field,
    val,
    path,
)
    return setfield_carray!(obj, field, val, path)
end

@inline function traced_setfield!(@nospecialize(obj::Dict), field, val, _path)
    return Base.setindex!(obj, field, val)
end

# fallback
@inline function setfield_carray!(obj, field, val, path)
    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[] && field == :data
        if val ∈ keys(DEBUG_BUFFER_POINTERS_STORE_DICT)
            if obj !== DEBUG_BUFFER_POINTERS_STORE_DICT[val]
                error("Aliased buffer cannot be assigned to multiple Concrete Structs. \
                       Path: $path.")
            end
        else
            DEBUG_BUFFER_POINTERS_STORE_DICT[val] = obj
        end
    end

    return Base.setproperty!(obj, field, val)
end

@inline function setfield_carray!(obj::ConcretePJRTArray, field, val, path)
    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[] && field == :data
        if val ∈ keys(DEBUG_BUFFER_POINTERS_STORE_DICT)
            if obj !== DEBUG_BUFFER_POINTERS_STORE_DICT[val]
                error("Aliased buffer cannot be assigned to multiple Concrete Structs. \
                       Path: $path.")
            end
        else
            DEBUG_BUFFER_POINTERS_STORE_DICT[val] = obj
        end
    end

    if field !== :data || typeof(val) == typeof(getfield(obj, field))
        return Base.setproperty!(obj, field, val)
    end

    # This case is triggered if the user had provided an unsharded input (NoSharding), but
    # we had to replicate it before feeding it to XLA
    @assert !Sharding.is_sharded(obj) "Expected unsharded input. Open an issue on \
                                       Reactant.jl with a MWE."
    devices = Reactant.XLA.device.(val)
    device = Reactant.XLA.device(only(getfield(obj, :data)))
    idx = findfirst(isequal(device), devices)
    return Base.setproperty!(obj, field, (val[idx],))
end

function traced_setfield_buffer!(runtime::Val, cache_dict, concrete_res, obj, field, path)
    return traced_setfield_buffer!(
        runtime, cache_dict, traced_getfield(obj, field), concrete_res, obj, field, path
    )
end

function traced_setfield_buffer!(::Val, _cache_dict, val, concrete_res, _obj, _field, path)
    return traced_setfield!(val, :data, concrete_res, path)
end

function traced_setfield_buffer!(
    ::Val{:PJRT},
    cache_dict,
    val::Union{TracedRArray,TracedRNumber},
    concrete_res,
    obj,
    field,
    path,
)
    if haskey(cache_dict, val)
        cval = cache_dict[val]
    else
        cval = if val isa TracedRArray
            ConcretePJRTArray{Reactant.unwrapped_eltype(val),ndims(val)}(
                concrete_res, size(val)
            )
        else
            ConcretePJRTNumber{Reactant.unwrapped_eltype(val)}(concrete_res)
        end
        cache_dict[val] = cval
    end
    return traced_setfield!(obj, field, cval, path)
end

function traced_setfield_buffer!(
    ::Val{:IFRT},
    cache_dict,
    val::Union{TracedRArray,TracedRNumber},
    concrete_res,
    obj,
    field,
    path,
)
    if haskey(cache_dict, val)
        cval = cache_dict[val]
    else
        cval = if val isa TracedRArray
            ConcreteIFRTArray{Reactant.unwrapped_eltype(val),ndims(val)}(
                concrete_res, size(val)
            )
        else
            ConcreteIFRTNumber{Reactant.unwrapped_eltype(val)}(concrete_res)
        end
        cache_dict[val] = cval
    end
    return traced_setfield!(obj, field, cval, path)
end

Base.@nospecializeinfer function create_result(
    @nospecialize(tocopy),
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
)
    if !isstructtype(typeof(tocopy))
        error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
    end

    T = Core.Typeof(tocopy)

    args = (
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        elems = Union{Symbol,Expr}[]

        for i in 1:fieldcount(T)
            # If the field is undefined we don't set it. A common example for this is `du2`
            # for Tridiagonal
            isdefined(tocopy, i) || continue
            ev = create_result(getfield(tocopy, i), append_path(path, i), args...)
            push!(elems, ev)
        end

        result = Expr(:new, T, elems...)

        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

Base.@nospecializeinfer function create_result(
    @nospecialize(tocopy::Enum),
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
)
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        result = Meta.quot(tocopy)

        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

function create_result(
    tocopy::ConcretePJRTNumber{T,D},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T,D}
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        @assert haskey(result_stores, path) "Expected $(path) in $(keys(result_stores))"
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
            if haskey(to_unreshard_results, path)
                error("TODO(#2234): Not yet Implemented. Use IFRT for this.")
            end
            sharding = pop!(path_to_shard_info, path)
            push!(used_shardinfo, sharding)
            result = :(ConcretePJRTNumber{$T}(($(restore)...,), $sharding))
        else
            result = :(ConcretePJRTNumber{$T}($restore))
        end
        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

function create_result(
    tocopy::ConcreteIFRTNumber{T},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T}
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        @assert haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
            if haskey(to_unreshard_results, path)
                error("TODO(#2234): Not yet Implemented.")
            end
            sharding = pop!(path_to_shard_info, path)
            push!(used_shardinfo, sharding)
            result = :(ConcreteIFRTNumber{$T}($(restore), $sharding))
        else
            result = :(ConcreteIFRTNumber{$T}($restore))
        end
        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

function create_result(
    tocopy::ConcretePJRTArray{T,N,D},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T,N,D}
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        @assert haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
            if haskey(to_unreshard_results, path)
                error("TODO(#2234): Not yet Implemented. Use IFRT for this.")
            end
            sharding = pop!(path_to_shard_info, path)
            push!(used_shardinfo, sharding)
            result =
                :(ConcretePJRTArray{$T,$N}(($(restore)...,), $(tocopy.shape), $sharding))
        else
            result = :(ConcretePJRTArray{$T,$N}($restore, $(tocopy.shape)))
        end
        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

function create_result(
    tocopy::ConcreteIFRTArray{T,N},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T,N}
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        @assert haskey(result_stores, path)
        restore = result_stores[path]
        delete!(result_stores, path)
        if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
            if haskey(to_unreshard_results, path)
                if !haskey(unresharded_arrays_cache, restore)
                    unresharded_array_sym = gensym(:unresharded_array)
                    push!(
                        unresharded_code,
                        :(
                            $unresharded_array_sym = generate_unresharded_ifrt_array(
                                $(restore),
                                $(to_unreshard_results[path][1]),
                                $(to_unreshard_results[path][2]),
                                global_mesh,
                            )
                        ),
                    )
                    unresharded_arrays_cache[restore] = unresharded_array_sym
                end
                return :(ConcreteIFRTArray{$T,$N}(
                    $(unresharded_arrays_cache[restore]), $(tocopy.shape)
                ))
            end
            sharding = pop!(path_to_shard_info, path)
            push!(used_shardinfo, sharding)
            result = :(ConcreteIFRTArray{$T,$N}($(restore), $(tocopy.shape), $sharding))
        else
            result = :(ConcreteIFRTArray{$T,$N}($(restore), $(tocopy.shape)))
        end
        push!(
            resultgen_code,
            quote
                $sym = $result
            end,
        )
        result_cache[tocopy] = sym
    end

    return result_cache[tocopy]
end

function generate_unresharded_ifrt_array(
    arr::Reactant.XLA.IFRT.AsyncArray, target_device, output_sharding, mesh
)
    size_arr = reverse(size(arr))

    single_device_arrays = Reactant.XLA.IFRT.disassemble_into_single_device_arrays(
        Reactant.XLA.IFRT.replicate_array_to_all_devices(
            arr, output_sharding, mesh, size_arr
        ),
        true,
    )
    devs = Reactant.XLA.device.(single_device_arrays)
    idx = findfirst(isequal(target_device), devs)
    @assert idx !== nothing
    res_arr = Reactant.XLA.IFRT.AsyncArray(single_device_arrays[idx], nothing)
    res_arr_size = reverse(size(res_arr))
    @assert size_arr == res_arr_size "Expected size of array to be $(size_arr), but got \
                                      $(res_arr_size)"

    ifrt_sharding = Reactant.XLA.sharding(res_arr)
    if !Reactant.XLA.IFRT.is_single_device_sharding(ifrt_sharding)
        error("Unexpected sharding of result array: $(string(ifrt_sharding))")
    end

    return res_arr
end

function create_result(
    tocopy::Array{T,N},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T,N}
    args = (
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        push!(
            resultgen_code,
            quote
                $sym = $(Array{T,N})(undef, $(size(tocopy)...,))
            end,
        )

        result_cache[tocopy] = sym

        for (i, v) in enumerate(tocopy)
            subexpr = create_result(v, append_path(path, i), args...)
            push!(
                resultgen_code,
                quote
                    @inbounds $sym[$i] = $subexpr
                end,
            )
        end
    end

    return result_cache[tocopy]
end

function create_result(
    tocopy::Tuple,
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
)
    args = (
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, append_path(path, k), args...))
    end
    return :(($(elems...),))
end

function create_result(
    tocopy::NamedTuple{K,T},
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {K,T}
    args = (
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )
    elems = Union{Symbol,Expr}[]
    for (i, (_, v)) in enumerate(pairs(tocopy))
        push!(elems, create_result(v, append_path(path, i), args...))
    end
    return :(NamedTuple{$K}(($(elems...),)))
end

function create_result(
    tocopy::D,
    @nospecialize(path::Tuple),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {K,V,D<:AbstractDict{K,V}}
    args = (
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        push!(
            resultgen_code,
            quote
                $sym = $D()
            end,
        )

        result_cache[tocopy] = sym

        for (k, v) in pairs(tocopy)
            subexpr = create_result(v, append_path(path, k), args...)
            # symbol keys must be quoted in generated code; otherwise
            # they are interpreted as variable references
            k_expr = k isa Symbol ? QuoteNode(k) : k
            push!(
                resultgen_code,
                quote
                    @inbounds $sym[$k_expr] = $subexpr
                end,
            )
        end
    end

    return quote
        $(result_cache[tocopy])
    end
end

function create_result(
    tocopy::Reactant.XLA.AbstractDevice,
    @nospecialize(_path::Tuple),
    _result_stores,
    _path_to_shard_info,
    _to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    _used_shardinfo,
    _result_cache,
    _var_idx,
    _resultgen_code,
)
    return Meta.quot(:($(tocopy)))
end

function create_result(
    tocopy::Union{Integer,AbstractFloat,AbstractString,Nothing,Type,Symbol,Char},
    @nospecialize(_path::Tuple),
    _result_stores,
    _path_to_shard_info,
    _to_unreshard_results,
    _unresharded_code::Vector{Expr},
    _unresharded_arrays_cache,
    _used_shardinfo,
    _result_cache,
    _var_idx,
    _resultgen_code,
)
    return Meta.quot(tocopy)
end

function assert_mismatched_sharding(
    sharding_from_input, hlo_sharding_from_executable::Reactant.XLA.HloSharding, size_x
)
    return assert_mismatched_sharding(
        Sharding.HloSharding(sharding_from_input, size_x).hlo_sharding,
        hlo_sharding_from_executable,
        size_x,
    )
end

function assert_mismatched_sharding(
    hlo_sharding_from_input::Reactant.XLA.HloSharding,
    hlo_sharding_from_executable::Reactant.XLA.HloSharding,
    _size_x,
)
    @assert hlo_sharding_from_executable == hlo_sharding_from_input "Sharding provided by the user ($(string(hlo_sharding_from_input))) does not match the sharding computed by XLA ($(string(hlo_sharding_from_executable))). This generally means that Reactant.jl made an error in generating the executable. Please open an issue with the error message and an MWE."
end

"""
    codegen_flatten!

Generate Julia code to extract the XLA buffers from input arguments.
The name is due to its similarity to the `flatten` function in `jax.tree_util.register_pytree_node`.

# Arguments

- `linear_args`: A list of arguments to be flattened.

# Returns

- `flatten_names`: A list of `Symbol`s representing the names of the flattened arguments.
- `flatten_code`: A list of `Expr`s to extract the XLA buffers from the input arguments.

# Note

The _linearized arguments_ do not directly refer to the arguments that have been flattened into a single list.
"""
function codegen_flatten!(
    linear_args,
    seen_args,
    is_sharded::Bool,
    linear_parameter_shardings,
    client,
    ndevices::Int,
)
    flatten_names = Symbol[]
    flatten_code = Expr[]
    runtime = XLA.runtime(client)
    resharded_inputs = Dict{Tuple,Any}()

    inv_seen_args = if is_sharded
        result = Reactant.OrderedIdDict()
        for (k, v) in seen_args
            result[v] = k
        end
        result
    else
        nothing
    end

    xla_parameter_sharding_sym = missing

    for (i, arg) in enumerate(linear_args)
        paths = (
            (
                p for
                p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
            )...,
        )
        path = if length(paths) == 1
            paths[1]
        else
            throw(
                "Invalid path duplication $(Reactant.TracedUtils.get_paths(arg)) into $(paths)",
            )
        end

        carg_sym = Symbol(:carg_, i)
        usbuf = Symbol(:usbuf_, i)

        flatcode = :(getindex(args, $(path[2])))
        for p in path[3:end]
            flatcode = :(traced_getfield($flatcode, $(Meta.quot(p))))
        end

        if runtime isa Val{:PJRT}
            push!(flatten_code, :($carg_sym = $flatcode))
            if is_sharded
                carg = inv_seen_args[arg]

                condensed_op_sharding = convert(
                    XLA.CondensedOpSharding, linear_parameter_shardings[i]
                )
                hlo_sharding_from_executable = convert(
                    XLA.HloSharding, condensed_op_sharding
                )
                if Sharding.is_sharded(carg)
                    # Check if the sharding provided is same as the one we have
                    assert_mismatched_sharding(
                        carg.sharding.sharding.hlo_sharding,
                        hlo_sharding_from_executable,
                        size(carg),
                    )

                    push!(flatten_code, :($usbuf = $carg_sym.data))
                    for j in 1:length(carg.sharding.mesh)
                        logical_id = carg.sharding.mesh.logical_device_ids[j]
                        sbuf = Symbol(:sbuf_, i, "_", logical_id)
                        push!(flatten_names, sbuf)
                        push!(
                            flatten_code,
                            :($sbuf = XLA.synced_buffer(getindex($usbuf, $(j)))),
                        )
                    end
                else
                    if DEBUG_DISABLE_RESHARDING[]
                        error("Resharding is disabled. Problematic input:\ntypeof: \
                               $(typeof(carg))\nsize: $(size(carg))\nsharding: \
                               $(carg.sharding)\nInput Index: $(i)\nInput Path: \
                               $(path[3:end])")
                    end

                    push!(flatten_code, :($usbuf = $carg_sym))
                    device_sym = gensym(:device)
                    push!(flatten_code, :($device_sym = Reactant.XLA.device($usbuf)))

                    if xla_parameter_sharding_sym === missing
                        xla_parameter_sharding_sym = :xla_parameter_sharding
                        pushfirst!(
                            flatten_code,
                            :(
                                $(xla_parameter_sharding_sym) = Reactant.XLA.get_parameter_shardings(
                                    thunk.exec
                                )
                            ),
                        )
                    end

                    resharded_inputs[path[3:end]] = (device_sym, xla_parameter_sharding_sym)

                    device_to_array_slices, _ = XLA.sharding_to_concrete_array_indices(
                        condensed_op_sharding, size(carg), 0:(ndevices - 1)
                    )

                    # Extract the buffer_slice
                    buf_slice = Dict{eltype(device_to_array_slices),Symbol}()
                    counter = 0
                    for j in 1:ndevices
                        sliced_buf = Symbol(:sliced_buf_, i, :_, counter)
                        slice = device_to_array_slices[j]
                        haskey(buf_slice, slice) && continue
                        counter += 1
                        push!(
                            flatten_code,
                            :(
                                $sliced_buf = only(
                                    Reactant._fast_slice($usbuf, $(slice...)).data
                                )
                            ),
                        )
                        buf_slice[slice] = sliced_buf
                    end

                    for j in 1:ndevices
                        buf = Symbol(:buf_, i, :_, j)
                        slice = device_to_array_slices[j]
                        sbuf = Symbol(:s, buf)
                        push!(flatten_names, sbuf)
                        push!(
                            flatten_code,
                            :(
                                $sbuf = XLA.copy_buffer_to_device(
                                    XLA.synced_buffer($(buf_slice[slice])),
                                    XLA.get_device(
                                        thunk.client, global_mesh.device_ids[$(j)]
                                    ),
                                )
                            ),
                        )
                    end
                end
            else
                push!(flatten_code, :($usbuf = $carg_sym.data))
                sbuf = Symbol(:sbuf_, i)
                push!(flatten_names, sbuf)
                if arg isa TracedRArray || arg isa TracedRNumber
                    push!(flatten_code, :($sbuf = only(XLA.synced_buffer($usbuf))))
                else
                    error("Unsupported type $(typeof(arg))")
                end
            end
            # Important to mark donated after we have extracted the data
            push!(
                flatten_code,
                :(donate_argument!(
                    donated_args_mask, $carg_sym, $i, donated_buffers, $(path)
                )),
            )
        elseif runtime isa Val{:IFRT}
            push!(flatten_code, :($carg_sym = $flatcode))
            push!(flatten_code, :($usbuf = $carg_sym.data))
            sbuf = Symbol(:sbuf_, i)
            push!(flatten_names, sbuf)
            if is_sharded
                carg = inv_seen_args[arg]

                condensed_op_sharding = convert(
                    XLA.CondensedOpSharding, linear_parameter_shardings[i]
                )

                if Sharding.is_sharded(carg)
                    # Check if the sharding provided is same as the one we have
                    assert_mismatched_sharding(
                        carg.sharding.sharding.hlo_sharding,
                        convert(XLA.HloSharding, condensed_op_sharding),
                        size(carg),
                    )

                    push!(flatten_code, :($sbuf = XLA.synced_buffer($usbuf)))
                else
                    if DEBUG_DISABLE_RESHARDING[]
                        error("Resharding is disabled. Problematic input:\ntypeof: \
                               $(typeof(carg))\nsize: $(size(carg))\nsharding: \
                               $(carg.sharding)\nInput Index: $(i)\nInput Path: \
                               $(path[3:end])")
                    end

                    device_sym = gensym(:device)
                    push!(flatten_code, :($device_sym = Reactant.XLA.device($usbuf)))

                    if xla_parameter_sharding_sym === missing
                        xla_parameter_sharding_sym = :xla_parameter_sharding
                        pushfirst!(
                            flatten_code,
                            :(
                                $(xla_parameter_sharding_sym) = Reactant.XLA.get_parameter_shardings(
                                    thunk.exec
                                )
                            ),
                        )
                    end

                    resharded_inputs[path] = (
                        device_sym, :($(xla_parameter_sharding_sym)[$(i)])
                    )

                    push!(
                        flatten_code,
                        :(
                            $(sbuf) = ifrt_resharded_buffer(
                                $(Reactant.unwrapped_eltype(carg)),
                                $(usbuf),
                                $(size(carg)),
                                thunk.client,
                                $(carg_sym).sharding,
                                thunk.global_device_ids,
                                $(xla_parameter_sharding_sym)[$(i)],
                            )
                        ),
                    )
                end
            else
                push!(flatten_code, :($sbuf = XLA.synced_buffer($usbuf)))
            end
            # Important to mark donated after we have extracted the data
            push!(
                flatten_code,
                :(donate_argument!(
                    donated_args_mask, $carg_sym, $i, donated_buffers, $(path)
                )),
            )
        else
            error("Unsupported runtime $runtime")
        end
    end

    # We reorder how the buffers are passed to the XLA call
    if is_sharded && runtime isa Val{:PJRT}
        flatten_names = vcat(eachrow(reshape(flatten_names, ndevices, :))...)
    end

    return flatten_names, flatten_code, resharded_inputs
end

function donate_argument!(
    donated_args_mask,
    carg::Union{ConcretePJRTNumber,ConcretePJRTArray},
    i::Int,
    donated_buffers,
    path,
)
    if donated_args_mask[i]
        buffers = Tuple(d.buffer for d in carg.data)
        if buffers in donated_buffers
            error("Donated buffer $(carg.data) is already marked as donated. Can't donate \
                   the same buffer multiple times. The argument is present at $(path)")
        end
        push!(donated_buffers, buffers)
        Reactant.mark_donated!(carg)
    end
end

function donate_argument!(
    donated_args_mask,
    carg::Union{ConcreteIFRTNumber,ConcreteIFRTArray},
    i::Int,
    donated_buffers,
    path,
)
    if donated_args_mask[i]
        if carg.data.buffer in donated_buffers
            error("Donated buffer $(carg.data) is already marked as donated. Can't donate \
                   the same buffer multiple times. The argument is present at $(path)")
        end
        push!(donated_buffers, carg.data.buffer)
        Reactant.mark_donated!(carg)
    end
end

# TODO(#2233): Currently we copy to host and then make the transfer to the sharded devices. This is
#      not ideal, we should be able to do a direct transfer using remapplan
function ifrt_resharded_buffer(
    ::Type{T}, ifrt_array, sz, client, reactant_sharding, global_device_ids, opsharding
) where {T}
    hlo_sharding = convert(XLA.HloSharding, opsharding)
    ifrt_sharding = XLA.IFRT.Sharding(
        XLA.get_device.((client,), global_device_ids), hlo_sharding
    )

    data = similar(Array{T,length(sz)}, sz)
    XLA.to_host(XLA.synced_buffer(ifrt_array), data, reactant_sharding)
    return XLA.IFRT.Array(client, data, ifrt_sharding)
end

"""
    codegen_unflatten!

Generate Julia code to wrap the XLA buffers back into the output result datatypes.
The name is due to its similarity to the `unflatten` function in `jax.tree_util.register_pytree_node`.
"""
function codegen_unflatten!(
    linear_args,
    preserved_args,
    concretized_res_names,
    linear_results,
    concrete_result,
    result_stores,
    path_to_shard_info,
    linear_result_shard_info,
    client,
    resharded_inputs,
)
    cache_dict = Symbol("cache_dict")
    needs_cache_dict = false
    unresharded_arrays_cache = Dict{Symbol,Symbol}()
    unresharded_code = Expr[]
    unflatten_code = Expr[]
    used_shardinfo = Set{Symbol}()

    runtime = XLA.runtime(client)
    if runtime isa Val{:PJRT}
        numtype = ConcretePJRTNumber
        arrtype = ConcretePJRTArray
    elseif runtime isa Val{:IFRT}
        numtype = ConcreteIFRTNumber
        arrtype = ConcreteIFRTArray
    else
        error("Unsupported runtime $runtime")
    end
    ctypes = Union{arrtype,numtype}

    to_unreshard_results = Dict{Tuple,Any}()

    # Ofcourse not thread-safe but this isn't meant for end-users anyways
    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]
        push!(unflatten_code, :(empty!(DEBUG_BUFFER_POINTERS_STORE_DICT)))
    end

    # mutate the result stores to point to the correct concrete results

    argprefix::Symbol = :args
    resprefix::Symbol = :result
    resargprefix::Symbol = :resargs

    for (concrete_res_name, result, shard_info) in
        zip(concretized_res_names, linear_results, linear_result_shard_info)
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == resprefix || p[1] == resargprefix)
            )...,
        )
        for path in paths
            if path[1] == resprefix
                unflatcode = :result
                path = path[2:end]

                if Reactant.TracedUtils.has_idx(result, argprefix)
                    argidx = Reactant.TracedUtils.get_idx(result, argprefix)
                    if haskey(resharded_inputs, argidx)
                        to_unreshard_results[path] = resharded_inputs[argidx]
                    end
                end

                result_stores[path] = concrete_res_name
                if path_to_shard_info !== nothing
                    path_to_shard_info[path] = shard_info
                end
                continue
            else
                @assert path[1] == resargprefix
                unflatcode = :(args[$(path[2])])

                need_to_unreshard = get(resharded_inputs, (:args, path[2:end]...), nothing)
                if need_to_unreshard !== nothing
                    @assert runtime isa Val{:IFRT} "PJRT is not supported here. Use IFRT \
                                                    instead."
                end

                path = path[3:end]
                for p in path[1:(end - 1)]
                    unflatcode = :(traced_getfield($unflatcode, $(Meta.quot(p))))
                end

                concrete_res_name_final = concrete_res_name
                if need_to_unreshard !== nothing
                    if !haskey(unresharded_arrays_cache, concrete_res_name)
                        unreshard_sym = gensym(:unresharded_buffer)
                        push!(
                            unresharded_code,
                            :(
                                $unreshard_sym = generate_unresharded_ifrt_array(
                                    $(concrete_res_name),
                                    $(need_to_unreshard[1]),
                                    $(need_to_unreshard[2]),
                                    global_mesh,
                                )
                            ),
                        )
                        unresharded_arrays_cache[concrete_res_name] = unreshard_sym
                    end
                    concrete_res_name_final = unresharded_arrays_cache[concrete_res_name]
                end

                if length(path) > 0
                    needs_cache_dict = true
                    # TODO(#2233): we might need to handle sharding here
                    unflatcode = quote
                        traced_setfield_buffer!(
                            $(runtime),
                            $(cache_dict),
                            $(concrete_res_name_final),
                            $(unflatcode),
                            $(Meta.quot(path[end])),
                            $(path),
                        )
                    end
                else
                    unflatcode = :(traced_setfield!(
                        $(unflatcode), :data, $(concrete_res_name_final), $(path)
                    ))
                end
                push!(unflatten_code, unflatcode)
            end
        end
    end

    if needs_cache_dict
        pushfirst!(
            unflatten_code,
            :($cache_dict = IdDict{Union{TracedRArray,TracedRNumber},$ctypes}()),
        )
    end

    result_cache = IdDict{Any,Symbol}()
    var_idx = Ref(0)
    resultgen_code = Expr[]

    for (result, arg_idx) in preserved_args
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :result)
            )...,
        )

        for path in paths
            arg = linear_args[arg_idx + 1]
            argpath = only((
                p for
                p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
            ))

            path = path[2:end]

            if in(path, keys(result_stores))
                continue
            end

            need_to_unreshard = get(resharded_inputs, (:args, argpath[2:end]...), nothing)
            if need_to_unreshard !== nothing
                # TODO(@avik-pal): I need an MWE to debug this codepath
                error("TODO(#2234): Not yet Implemented. Open an issue on Reactant.jl.")
            end

            argres = :(args[$(argpath[2])])
            for p in argpath[3:end]
                argres = :(traced_getfield($argres, $(Meta.quot(p))))
            end

            sym = Symbol("result", var_idx[])
            var_idx[] += 1

            push!(
                resultgen_code,
                quote
                    $sym = $argres.data
                end,
            )

            result_stores[path] = sym
        end
    end

    result_code = create_result(
        concrete_result,
        (),
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code,
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    # if some argument is mutated, change them to point to the correct concrete results
    orig_buffer_available = trues(length(linear_args))
    for (pi, (result, arg_idx)) in enumerate(preserved_args)
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :resargs || p[1] == :args)
            )...,
        )

        arg = linear_args[arg_idx + 1]
        argpath = only((
            p for p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
        ))
        argpath_value = nothing

        if any(parg -> parg[2] == arg_idx, preserved_args)
            orig_buffer_available[arg_idx + 1] = false
        end

        for path in paths
            @assert path[1] == :resargs || path[1] == :args "Expected :resargs or :args, got $(path[1])"
            initial_path = path
            # We can optimize cases where we set the arg to itself
            if path[2:end] == argpath[2:end]
                continue
            end
            res = :(args[$(path[2])])

            path = path[3:end]
            for p in path
                res = :(traced_getfield($res, $(Meta.quot(p))))
            end

            need_to_unreshard = get(resharded_inputs, (:args, argpath[2:end]...), nothing)
            if need_to_unreshard !== nothing
                # TODO(@avik-pal): I need an MWE to debug this codepath
                error("TODO(#2234): Not yet Implemented. Open an issue on Reactant.jl.")
            end

            if isnothing(argpath_value)
                # this traced array take an identity value derived from another argument but
                # is its own traced array. As such, it needs to allocate a new buffer instead of using the arg directly.
                needs_copy =
                    initial_path[1] === :args &&
                    argpath[1] === :args &&
                    !orig_buffer_available[arg_idx + 1]
                orig_buffer_available[arg_idx + 1] = false

                argres = :(args[$(argpath[2])])
                for p in argpath[3:end]
                    argres = :(traced_getfield($argres, $(Meta.quot(p))))
                end
                argpath_value = Symbol(:argpath_value, pi)
                argres_data = :($(argres).data)
                if needs_copy
                    if runtime isa Val{:PJRT}
                        argres_data = :(map(copy, $argres_data))
                    elseif runtime isa Val{:IFRT}
                        argres_data = :(copy($argres_data))
                    else
                        error("unknown runtime $runtime")
                    end
                end
                push!(unflatten_code, :($argpath_value = $argres_data))
            end

            res = :(traced_setfield!($res, :data, $argpath_value, $(path)))
            push!(unflatten_code, res)
        end
    end

    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]
        push!(unflatten_code, :(empty!(DEBUG_BUFFER_POINTERS_STORE_DICT)))
    end

    # generate return object which stores the concrete results in some arbitrary way
    return (
        Expr[
            unresharded_code...,
            resultgen_code...,
            :(result = $result_code),
            unflatten_code...,
        ],
        used_shardinfo,
    )
end

"""
    codegen_xla_call

Generate Julia code to call the XLA executable.

# Arguments

- `flatten_names`: A list of `Symbol`s representing the names of the flattened linear arguments.
- `nresults`: The number of results to expect.
- `is_pure`: Whether the function being compiled is pure (i.e., has no side effects)
"""
function codegen_xla_call(
    flatten_names, nresults, is_sharded::Bool, ndevices::Int, is_pure::Bool
)
    flatten_buffer_refs = map(n -> :($n.buffer), flatten_names)

    base_symbol_name = is_sharded ? Symbol(:result_buffer_m, ndevices, :_) : :result_buffer_
    concretized_res_names = Symbol[Symbol(base_symbol_name, i) for i in 1:nresults]
    concretized_res_code = map(enumerate(concretized_res_names)) do (i, varname)
        :($varname = linearized_results[$i])
    end

    xla_call_code = if nresults == 0 && is_pure
        :()
    else
        if is_sharded
            quote
                GC.@preserve $(flatten_names...) begin
                    linearized_results = XLA.execute(
                        thunk.exec,
                        ($(flatten_buffer_refs...),),
                        UInt8.(Tuple(donated_args_mask)),
                        Val($nresults),
                        Val($ndevices),
                    )
                end
                $(concretized_res_code...)
            end
        else
            quote
                GC.@preserve $(flatten_names...) begin
                    linearized_results = XLA.execute_sharded(
                        thunk.exec,
                        thunk.device,
                        ($(flatten_buffer_refs...),),
                        UInt8.(Tuple(donated_args_mask)),
                        Val($nresults),
                    )
                end
                $(concretized_res_code...)
            end
        end
    end

    return concretized_res_names, xla_call_code
end

# generate the shard info code. we cannot embed any mesh information into the code here,
# else serialization will be incorrect.
function codegen_shard_info(
    is_sharded, nresults::Int, linear_results, output_reactant_shardings, exec, ndevices
)
    !is_sharded && return Expr[], Expr[], [nothing for _ in 1:nresults]

    shard_info_code = Expr[]
    optional_shard_info_code = Expr[]
    output_hlo_shardings_var = missing
    output_hlo_shardings = XLA.get_output_shardings(exec)

    linear_result_shard_info = Vector{Symbol}(undef, length(linear_results))
    mesh_codegen_cache = Dict{Tuple,Symbol}()

    for i in 1:nresults
        res_size = size(linear_results[i])
        array_slices, hlo_sharding = XLA.compute_array_indices_and_hlo_sharding(
            output_hlo_shardings[i],
            res_size,
            0:(ndevices - 1), # verify if this is correct?
        )

        if output_reactant_shardings !== missing
            reactant_sharding = output_reactant_shardings[i]
            use_hlo_sharding =
                reactant_sharding isa Sharding.NoSharding ||
                Sharding.HloSharding(reactant_sharding, res_size).hlo_sharding !=
                hlo_sharding
        else
            use_hlo_sharding = true
        end

        var_name = Symbol(:shard_info_, i)
        if use_hlo_sharding
            if output_hlo_shardings_var === missing
                output_hlo_shardings_var = :output_hlo_shardings
                pushfirst!(
                    shard_info_code,
                    :(
                        $(output_hlo_shardings_var) = Reactant.XLA.get_output_shardings(
                            thunk.exec
                        )
                    ),
                )
            end

            push!(
                optional_shard_info_code,
                :(
                    $(var_name) = Sharding.ShardInfo(
                        Sharding.HloSharding(
                            convert(XLA.HloSharding, $(output_hlo_shardings_var)[$(i)]),
                            global_mesh,
                            $(ntuple(Returns(true), length(res_size))),
                            $(ntuple(Returns(-1), length(res_size))),
                            nothing,
                        ),
                        $(array_slices),
                    )
                ),
            )
        else
            mesh = output_reactant_shardings[i].mesh
            mesh_key = (mesh.logical_device_ids, mesh.axis_names, mesh.axis_sizes)
            if haskey(mesh_codegen_cache, mesh_key)
                mesh_name = mesh_codegen_cache[mesh_key]
            else
                mesh_name = gensym(:mesh)
                push!(
                    shard_info_code,
                    :(
                        $(mesh_name) = Sharding.Mesh(
                            thunk.global_device_ids,
                            $(mesh.logical_device_ids),
                            $(mesh.axis_names),
                            $(mesh.axis_sizes),
                        )
                    ),
                )
                mesh_codegen_cache[mesh_key] = mesh_name
            end

            new_sharding = Sharding.codegen_with_new_mesh(
                output_reactant_shardings[i], mesh_name
            )
            push!(
                optional_shard_info_code,
                :($(var_name) = Sharding.ShardInfo($(new_sharding), $(array_slices))),
            )
        end
        linear_result_shard_info[i] = var_name
    end
    return shard_info_code, optional_shard_info_code, linear_result_shard_info
end
