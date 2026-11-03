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

function check_aliased_buffer_assignment(
    @nospecialize(obj), field, val, path; buffer_field=:data
)
    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[] && field == buffer_field
        if val ∈ keys(DEBUG_BUFFER_POINTERS_STORE_DICT)
            if obj !== DEBUG_BUFFER_POINTERS_STORE_DICT[val]
                error("Aliased buffer cannot be assigned to multiple Concrete Structs. \
                       Path: $path.")
            end
        else
            DEBUG_BUFFER_POINTERS_STORE_DICT[val] = obj
        end
    end
end

@inline function traced_getfield(@nospecialize(obj::Dict), field)
    return Base.getindex(obj, field)
end

@inline function traced_getfield(@nospecialize(obj), field)
    return Base.getfield(obj, field)
end

# A path can address the payload inside a `TracedEnum` while the object actually present at
# the enum's position (e.g. an untraced branch counterpart) is still the plain enum; there
# is nothing to descend into, and callers skip untraced targets.
@inline function traced_getfield(@nospecialize(obj::Base.Enum), field)
    # Only the synthetic payload field can stand in for a plain enum.
    field === 1 && return obj
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
    @nospecialize(obj::Union{AbstractConcreteArray,AbstractConcreteNumber}), field
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
    check_aliased_buffer_assignment(obj, field, val, path)
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
    check_aliased_buffer_assignment(obj, field, val, path)
    return Base.setproperty!(obj, field, val)
end

@inline function setfield_carray!(obj::ConcretePJRTArray, field, val, path)
    check_aliased_buffer_assignment(obj, field, val, path)

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

# A captured traced enum must be replaced in its containing field: its typed payload
# cannot accept a concrete number. Other values retain the usual payload write-back.
function traced_setfield_buffer_at_parent!(
    runtime, cache_dict, concrete_res, parent, field, payload_field, path
)
    obj = traced_getfield(parent, field)
    if obj isa Reactant.TracedEnum
        if haskey(cache_dict, obj)
            concrete_enum = cache_dict[obj]
        else
            payload = obj.value
            if haskey(cache_dict, payload)
                concrete_payload = cache_dict[payload]
            else
                T = Reactant.unwrapped_eltype(payload)
                concrete_payload = if runtime isa Val{:PJRT}
                    ConcretePJRTNumber{T}(concrete_res)
                else
                    ConcreteIFRTNumber{T}(concrete_res)
                end
                cache_dict[payload] = concrete_payload
            end
            E = typeof(obj).parameters[1]
            concrete_enum = Reactant.ConcreteEnum{E}(concrete_payload)
            cache_dict[obj] = concrete_enum
        end
        return traced_setfield!(parent, field, concrete_enum, path)
    end
    return traced_setfield_buffer!(
        runtime, cache_dict, concrete_res, obj, payload_field, path
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
                    unresharded_array_sym = Symbol(
                        :unresharded_array_, length(unresharded_arrays_cache) + 1
                    )
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

function add_chunk_param!(params::Vector{Pair{Symbol,Any}}, @nospecialize(val))
    sym = Symbol(:_p_, length(params) + 1)
    push!(params, sym => val)
    return sym
end

# Hoist per-buffer runtime values (buffer slot indices, array shapes, sharding slices/specs,
# and runtime container indices marked with `Base.identity`) out of an element's generated
# AST into `params`, so elements of the same structural shape share one `ThunkChunk`.
function hoist_chunk_params!(@nospecialize(ex), params::Vector{Pair{Symbol,Any}})
    ex isa Expr || return ex
    if ex.head === :quote || ex.head === :meta
        return ex
    elseif ex.head === :ref &&
        length(ex.args) == 2 &&
        (
            ex.args[1] === :linearized_results ||
            ex.args[1] === :output_hlo_shardings ||
            ex.args[1] === :aliased_arg_buffers ||
            ex.args[1] === :preserved_result_buffers
        ) &&
        ex.args[2] isa Int
        p = add_chunk_param!(params, ex.args[2])
        return Expr(:ref, ex.args[1], p)
    elseif ex.head === :call
        callee = ex.args[1]
        if callee === Base.identity && length(ex.args) == 2
            arg = ex.args[2]
            val = arg isa QuoteNode ? arg.value : arg
            return add_chunk_param!(params, val)
        elseif callee isa Expr &&
            callee.head === :curly &&
            (
                callee.args[1] === :ConcretePJRTArray ||
                callee.args[1] === :ConcreteIFRTArray
            ) &&
            length(ex.args) >= 3 &&
            ex.args[3] isa Tuple
            new_args = Any[callee, hoist_chunk_params!(ex.args[2], params)]
            push!(new_args, add_chunk_param!(params, ex.args[3]))
            for k in 4:length(ex.args)
                push!(new_args, hoist_chunk_params!(ex.args[k], params))
            end
            return Expr(:call, new_args...)
        elseif callee == :(Sharding.ShardInfo) && length(ex.args) == 3
            return Expr(
                :call,
                callee,
                hoist_chunk_params!(ex.args[2], params),
                add_chunk_param!(params, ex.args[3]),
            )
        elseif callee === Sharding.NamedSharding && length(ex.args) == 6
            return Expr(
                :call,
                callee,
                ex.args[2],
                add_chunk_param!(params, ex.args[3]),
                add_chunk_param!(params, ex.args[4]),
                add_chunk_param!(params, ex.args[5]),
                add_chunk_param!(params, ex.args[6]),
            )
        end
    elseif ex.head === :macrocall
        new_args = Any[ex.args[1], ex.args[2]]
        for k in 3:length(ex.args)
            push!(new_args, hoist_chunk_params!(ex.args[k], params))
        end
        return Expr(:macrocall, new_args...)
    end
    return Expr(ex.head, Any[hoist_chunk_params!(a, params) for a in ex.args]...)
end

function hoist_scalar_element!(@nospecialize(subexpr), params::Vector{Pair{Symbol,Any}})
    if subexpr isa Union{Integer,AbstractFloat,AbstractString,Nothing,Char}
        return add_chunk_param!(params, subexpr)
    elseif subexpr isa QuoteNode && subexpr.value isa Symbol
        return add_chunk_param!(params, subexpr.value)
    end
    return subexpr
end

function emit_indexed_element_chunk!(
    resultgen_code,
    container_sym::Symbol,
    @nospecialize(idx_val),
    elem_code::Vector{Expr},
    @nospecialize(subexpr),
    reorderable::Bool,
)
    if any(is_thunk_chunk, elem_code)
        for stmt in elem_code
            push!(resultgen_code, is_thunk_chunk(stmt) ? stmt : thunk_chunk(stmt))
        end
        params = Pair{Symbol,Any}[:_elem_idx => idx_val]
        val_expr = hoist_scalar_element!(subexpr, params)
        push!(
            resultgen_code,
            thunk_chunk(
                :(@inbounds $container_sym[_elem_idx] = $val_expr); params, reorderable
            ),
        )
    else
        params = Pair{Symbol,Any}[:_elem_idx => idx_val]
        flat_stmts = flatten_statements(elem_code)
        hoisted_stmts = Any[hoist_chunk_params!(s, params) for s in flat_stmts]
        val_expr = if isempty(hoisted_stmts)
            hoist_scalar_element!(subexpr, params)
        else
            hoist_chunk_params!(subexpr, params)
        end
        push!(hoisted_stmts, :(@inbounds $container_sym[_elem_idx] = $val_expr))
        push!(resultgen_code, thunk_chunk(hoisted_stmts...; params, reorderable))
    end
    return nothing
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
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        push!(
            resultgen_code, thunk_chunk(:($sym = $(Array{T,N})(undef, $(size(tocopy)...,))))
        )

        result_cache[tocopy] = sym

        for (i, v) in enumerate(tocopy)
            elem_code = Expr[]
            sub_args = (
                result_stores,
                path_to_shard_info,
                to_unreshard_results,
                unresharded_code::Vector{Expr},
                unresharded_arrays_cache,
                used_shardinfo,
                result_cache,
                var_idx,
                elem_code,
            )
            subexpr = create_result(v, append_path(path, i), sub_args...)
            emit_indexed_element_chunk!(resultgen_code, sym, i, elem_code, subexpr, true)
        end
    end

    return result_cache[tocopy]
end

function _is_small_typeassert(@nospecialize(T::Type))
    isconcretetype(T) || return false
    ismutabletype(T) && return true
    return sizeof(T) <= 256
end

function _has_no_huge_inline_fields(@nospecialize(obj))
    T = Core.Typeof(obj)
    isconcretetype(T) || return false
    (!ismutabletype(T) && sizeof(T) > 256) && return false
    for i in 1:fieldcount(T)
        FT = fieldtype(T, i)
        if isconcretetype(FT) && !ismutabletype(FT) && sizeof(FT) > 256
            return false
        end
    end
    return true
end

@noinline function unpack_tuple_to_any_vec(@nospecialize(t::Tuple))
    n = nfields(t)
    out = Vector{Any}(undef, n)
    for i in 1:n
        @inbounds out[i] = getfield(t, i)
    end
    return out
end

@noinline function pack_vec_to_tuple_or_nt(@nospecialize(T::Type), elems::Vector{Any})
    if T <: Tuple
        return Core._apply_iterate(Base.iterate, tuple, elems)
    end
    return GC.@preserve elems ccall(
        :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, elems, UInt32(length(elems))
    )
end

@noinline function pack_homogeneous_vec(@nospecialize(T::Type), @nospecialize(v::Vector))
    T <: Tuple && return Tuple(v)
    n = length(v)
    elems = Vector{Any}(undef, n)
    for i in 1:n
        @inbounds elems[i] = v[i]
    end
    return GC.@preserve elems ccall(
        :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, elems, UInt32(n)
    )
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
    len = length(tocopy)
    len == 0 && return :()

    if len <= 4 && _has_no_huge_inline_fields(tocopy)
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
        elems = Any[create_result(tocopy[i], append_path(path, i), args...) for i in 1:len]
        return :(($(elems...),))
    end

    TT = typeof(tocopy)
    ET = fieldtype(TT, 1)
    is_homog =
        allequal(fieldtypes(TT)) &&
        isconcretetype(ET) &&
        (ismutabletype(ET) || sizeof(ET) <= 256)
    buf_sym = Symbol("result", var_idx[])
    var_idx[] += 1
    sym = Symbol("result", var_idx[])
    var_idx[] += 1
    buf_type = is_homog ? Vector{ET} : Vector{Any}
    push!(resultgen_code, thunk_chunk(:($buf_sym = $buf_type(undef, $len))))
    for (i, v) in enumerate(tocopy)
        elem_code = Expr[]
        sub_args = (
            result_stores,
            path_to_shard_info,
            to_unreshard_results,
            unresharded_code::Vector{Expr},
            unresharded_arrays_cache,
            used_shardinfo,
            result_cache,
            var_idx,
            elem_code,
        )
        subexpr = create_result(v, append_path(path, i), sub_args...)
        emit_indexed_element_chunk!(resultgen_code, buf_sym, i, elem_code, subexpr, true)
    end
    pack_fn = is_homog ? :pack_homogeneous_vec : :pack_vec_to_tuple_or_nt
    pack_expr = :($pack_fn($TT, $buf_sym))
    if isempty(path) && _is_small_typeassert(TT)
        pack_expr = :($pack_expr::$TT)
    end
    push!(resultgen_code, thunk_chunk(:($sym = $pack_expr)))
    return sym
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
    len = length(tocopy)
    if len == 0
        return :(NamedTuple{$K}(()))
    end

    if len <= 4 && _has_no_huge_inline_fields(tocopy)
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
        elems = Any[
            create_result(v, append_path(path, i), args...) for
            (i, (_, v)) in enumerate(pairs(tocopy))
        ]
        return :(NamedTuple{$K}(($(elems...),)))
    end

    NTT = NamedTuple{K,T}
    ET = fieldtype(T, 1)
    is_homog =
        allequal(fieldtypes(T)) &&
        isconcretetype(ET) &&
        (ismutabletype(ET) || sizeof(ET) <= 256)
    buf_sym = Symbol("result", var_idx[])
    var_idx[] += 1
    sym = Symbol("result", var_idx[])
    var_idx[] += 1
    buf_type = is_homog ? Vector{ET} : Vector{Any}
    push!(resultgen_code, thunk_chunk(:($buf_sym = $buf_type(undef, $len))))
    for (i, (_, v)) in enumerate(pairs(tocopy))
        elem_code = Expr[]
        sub_args = (
            result_stores,
            path_to_shard_info,
            to_unreshard_results,
            unresharded_code::Vector{Expr},
            unresharded_arrays_cache,
            used_shardinfo,
            result_cache,
            var_idx,
            elem_code,
        )
        subexpr = create_result(v, append_path(path, i), sub_args...)
        emit_indexed_element_chunk!(resultgen_code, buf_sym, i, elem_code, subexpr, true)
    end
    pack_fn = is_homog ? :pack_homogeneous_vec : :pack_vec_to_tuple_or_nt
    pack_expr = :($pack_fn($NTT, $buf_sym))
    if isempty(path) && _is_small_typeassert(NTT)
        pack_expr = :($pack_expr::$NTT)
    end
    push!(resultgen_code, thunk_chunk(:($sym = $pack_expr)))
    return sym
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
    if !haskey(result_cache, tocopy)
        sym = Symbol("result", var_idx[])
        var_idx[] += 1

        push!(resultgen_code, thunk_chunk(:($sym = $D())))

        result_cache[tocopy] = sym
        reorderable = tocopy isa Dict

        for (k, v) in pairs(tocopy)
            elem_code = Expr[]
            sub_args = (
                result_stores,
                path_to_shard_info,
                to_unreshard_results,
                unresharded_code::Vector{Expr},
                unresharded_arrays_cache,
                used_shardinfo,
                result_cache,
                var_idx,
                elem_code,
            )
            subexpr = create_result(v, append_path(path, k), sub_args...)
            emit_indexed_element_chunk!(
                resultgen_code, sym, k, elem_code, subexpr, reorderable
            )
        end
    end

    return result_cache[tocopy]
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
    # The metadata is ignored on purpose: XLA attaches things like the shardy reduction op
    # to the sharding it computes, which doesn't change how the data is laid out.
    @assert Reactant.XLA.isequal_ignoring_metadata(
        hlo_sharding_from_executable, hlo_sharding_from_input
    ) "Sharding provided by the user ($(string(hlo_sharding_from_input))) does not match the sharding computed by XLA ($(string(hlo_sharding_from_executable))). This generally means that Reactant.jl made an error in generating the executable. Please open an issue with the error message and an MWE."
end

"""
    is_runtime_index(obj, field)

Whether `traced_getfield(obj, field)` indexes into a container whose element type does not
depend on `field` (an `Array`, `Dict`, or non-tiny homogeneous `Tuple`/`NamedTuple`), in
which case `field` can be passed as a runtime parameter, as opposed to accessing a
struct/heterogeneous-tuple field where `field` must remain a compile-time constant for type
inference. Mirrors the dispatch of [`traced_getfield`](@ref).
"""
function is_runtime_index(@nospecialize(obj), @nospecialize(field))
    obj isa Dict && return true
    obj isa Base.Enum && return false
    field isa Integer || return false
    if obj isa Union{Tuple,NamedTuple}
        length(obj) >= 4 && allequal(fieldtypes(typeof(obj))) || return false
        ET = fieldtype(typeof(obj), 1)
        return isconcretetype(ET) && (ismutabletype(ET) || sizeof(ET) <= 256)
    end
    obj isa Union{AbstractConcreteArray,AbstractConcreteNumber} && return false
    if obj isa AbstractArray
        T = eltype(obj)
        if T <: Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}
            return obj isa Array
        end
        return !(isbitstype(T) || ancestor(obj) isa RArray || obj isa AbstractRange)
    end
    return false
end

@noinline function unpack_homogeneous_vec(@nospecialize(obj), ::Type{ET}) where {ET}
    n = nfields(obj)
    out = Vector{ET}(undef, n)
    for i in 1:n
        @inbounds out[i] = getfield(obj, i)::ET
    end
    return out
end

@noinline function unpack_arg_homogeneous_vec(
    @nospecialize(args), idx::Int, ::Type{ET}
) where {ET}
    return unpack_homogeneous_vec(getfield(args, idx), ET)
end

@noinline function unpack_selected_fields(@nospecialize(obj), fields::Tuple)
    n = length(fields)
    out = Vector{Any}(undef, n)
    for i in 1:n
        @inbounds out[i] = traced_getfield(obj, fields[i])
    end
    return out
end

@noinline function unpack_arg_selected_fields(@nospecialize(args), idx::Int, fields::Tuple)
    return unpack_selected_fields(traced_getfield(args, idx), fields)
end

# Safely traverse a path step on `roots` at compile time; returns `nothing` if an untyped
# mutable slot held a placeholder (e.g. `nothing`) prior to being overwritten during tracing.
function _safe_traced_getfield(@nospecialize(obj), @nospecialize(field))
    if obj isa Dict
        return haskey(obj, field) ? obj[field] : nothing
    elseif obj isa Base.Enum
        field === 1 && return obj
        return if field isa Symbol && hasfield(typeof(obj), field)
            getfield(obj, field)
        else
            nothing
        end
    elseif obj isa Array{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}}
        return checkbounds(Bool, obj, field) ? obj[field] : nothing
    elseif obj isa AbstractArray
        T_el = eltype(typeof(obj))
        if obj isa
           AbstractArray{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}} ||
            isbitstype(T_el) ||
            ancestor(obj) isa RArray ||
            obj isa AbstractRange
            return if field isa Symbol && hasfield(typeof(obj), field)
                getfield(obj, field)
            else
                nothing
            end
        end
        return checkbounds(Bool, obj, field) ? obj[field] : nothing
    elseif obj isa Union{AbstractConcreteArray,AbstractConcreteNumber}
        return if field isa Symbol && hasproperty(obj, field)
            getproperty(obj, field)
        else
            nothing
        end
    elseif field isa Symbol
        return hasfield(typeof(obj), field) ? getfield(obj, field) : nothing
    elseif field isa Int
        return (1 <= field <= nfields(obj)) ? getfield(obj, field) : nothing
    end
    return nothing
end

# Statically check whether `steps` starting from `T = typeof(roots)` is guaranteed by
# container field/element types to resolve to an `AbstractConcreteArray` or
# `AbstractConcreteNumber` (so `:resargs` write-back never sees a `TracedRArray`,
# `TracedRNumber`, or `TracedEnum`, and does not need `cache_dict`).
function _is_statically_concrete_target(@nospecialize(T::Type), steps::Tuple)
    for p in steps
        if T isa Union || T === Any || T === Union{} || isabstracttype(T)
            return false
        elseif T <: Tuple || T <: NamedTuple
            if p isa Int && 1 <= p <= fieldcount(T)
                T = fieldtype(T, p)
            elseif p isa Symbol && hasfield(T, p)
                T = fieldtype(T, p)
            else
                return false
            end
        elseif T <: Array{<:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}}
            T = eltype(T)
        elseif T <: AbstractArray
            T_el = eltype(T)
            if T <: AbstractArray{
                    <:Union{ConcretePJRTNumber,ConcreteIFRTNumber,TracedRNumber}
                } ||
                isbitstype(T_el) ||
                T <: RArray ||
                T <: AbstractRange
                if p isa Symbol && hasfield(T, p)
                    T = fieldtype(T, p)
                else
                    return false
                end
            else
                T = T_el
            end
        elseif T <: Dict
            T = valtype(T)
        elseif T <: Union{AbstractConcreteArray,AbstractConcreteNumber}
            if p isa Symbol && hasfield(T, p)
                T = fieldtype(T, p)
            else
                return false
            end
        elseif isstructtype(T)
            if p isa Symbol && hasfield(T, p)
                T = fieldtype(T, p)
            elseif p isa Int && 1 <= p <= fieldcount(T)
                T = fieldtype(T, p)
            else
                return false
            end
        else
            return false
        end
    end
    return (T <: AbstractConcreteArray || T <: AbstractConcreteNumber) && !(T === Union{})
end

struct ArgPathEntry
    prefix_steps::Tuple
    suffix_steps::Tuple
    payload::Any
end

function _walk_arg_path_group!(
    out_stmts::Vector{Expr},
    @nospecialize(obj),
    obj_expr,
    entries::Vector{ArgPathEntry},
    depth::Int,
    split_counter::Base.RefValue{Int},
    emit_leaf!::F;
    arg_root_idx::Int=0,
    from_any_vec::Bool=false,
    parent_param::Union{Nothing,Pair{Symbol,<:Any}}=nothing,
) where {F}
    deeper = ArgPathEntry[]
    for e in entries
        if length(e.prefix_steps) < depth
            params = Pair{Symbol,Any}[]
            parent_param !== nothing && push!(params, parent_param)
            leaf_expr =
                if from_any_vec && obj !== nothing && _is_small_typeassert(Core.Typeof(obj))
                    :($obj_expr::$(Core.Typeof(obj)))
                else
                    obj_expr
                end
            _, suffix_codes = codegen_getfield_path(
                obj, leaf_expr, e.suffix_steps, params, :sidx_
            )
            emit_leaf!(out_stmts, leaf_expr, suffix_codes, params, e.payload)
        else
            push!(deeper, e)
        end
    end
    isempty(deeper) && return nothing

    if is_runtime_index(obj, deeper[1].prefix_steps[depth])
        base_expr = if obj isa Union{Tuple,NamedTuple}
            ET = fieldtype(typeof(obj), 1)
            split_counter[] += 1
            hvec_sym = Symbol(:_arg_hvec_, split_counter[])
            unpack_call = if arg_root_idx > 0
                :(unpack_arg_homogeneous_vec(args, $arg_root_idx, $ET))
            else
                :(unpack_homogeneous_vec($obj_expr, $ET))
            end
            push!(out_stmts, :($hvec_sym = $unpack_call))
            hvec_sym
        elseif from_any_vec && obj !== nothing && _is_small_typeassert(Core.Typeof(obj))
            :($obj_expr::$(Core.Typeof(obj)))
        else
            obj_expr
        end
        for e in deeper
            params = Pair{Symbol,Any}[]
            parent_param !== nothing && push!(params, parent_param)
            rem_prefix = e.prefix_steps[depth:end]
            target_expr, _ = codegen_getfield_path(
                obj, base_expr, rem_prefix, params, :idx_
            )
            target_obj = obj
            for p in rem_prefix
                target_obj = _safe_traced_getfield(target_obj, p)
            end
            _, suffix_codes = codegen_getfield_path(
                target_obj, target_expr, e.suffix_steps, params, :sidx_
            )
            emit_leaf!(out_stmts, target_expr, suffix_codes, params, e.payload)
        end
        return nothing
    end

    distinct_fields = Any[]
    groups = Dict{Any,Vector{ArgPathEntry}}()
    for e in deeper
        f = e.prefix_steps[depth]
        grp = get!(groups, f) do
            push!(distinct_fields, f)
            return ArgPathEntry[]
        end
        push!(grp, e)
    end

    M = length(distinct_fields)
    if (M <= 4 && _has_no_huge_inline_fields(obj)) || obj === nothing
        base_expr =
            if from_any_vec && obj !== nothing && _is_small_typeassert(Core.Typeof(obj))
                :($obj_expr::$(Core.Typeof(obj)))
            else
                obj_expr
            end
        for f in distinct_fields
            child_obj = _safe_traced_getfield(obj, f)
            child_expr = :(traced_getfield($base_expr, $(Meta.quot(f))))
            next_root_idx = (depth == 1 && f isa Int) ? f : 0
            _walk_arg_path_group!(
                out_stmts,
                child_obj,
                child_expr,
                groups[f],
                depth + 1,
                split_counter,
                emit_leaf!;
                arg_root_idx=next_root_idx,
                from_any_vec=false,
                parent_param=parent_param,
            )
        end
        return nothing
    end

    fields_tup = Tuple(distinct_fields)
    split_counter[] += 1
    vec_sym = Symbol(:_arg_vec_, split_counter[])
    unpack_call = if arg_root_idx > 0
        :(unpack_arg_selected_fields(args, $arg_root_idx, $(QuoteNode(fields_tup))))
    else
        :(unpack_selected_fields($obj_expr, $(QuoteNode(fields_tup))))
    end
    push!(out_stmts, :($vec_sym = $unpack_call))

    for (j, f) in enumerate(distinct_fields)
        grp = groups[f]
        child_obj = _safe_traced_getfield(obj, f)
        all_direct_leaves = all(e -> length(e.prefix_steps) == depth, grp)
        if all_direct_leaves && parent_param === nothing
            _walk_arg_path_group!(
                out_stmts,
                child_obj,
                :(getindex($vec_sym, _vidx)),
                grp,
                depth + 1,
                split_counter,
                emit_leaf!;
                arg_root_idx=0,
                from_any_vec=true,
                parent_param=Pair{Symbol,Any}(:_vidx, j),
            )
        else
            _walk_arg_path_group!(
                out_stmts,
                child_obj,
                :(getindex($vec_sym, $j)),
                grp,
                depth + 1,
                split_counter,
                emit_leaf!;
                arg_root_idx=0,
                from_any_vec=true,
                parent_param=parent_param,
            )
        end
    end
    return nothing
end

# Traverse `entries` rooted at `args` / `roots`, unpacking any wide (> 4 field) containers
# into heap `Vector`s via `@nospecialize` helpers so outlined `ThunkChunk`s and loop bodies
# never unbox or pass `O(N)`-pointer inline aggregates, while leaving small (`<= 4` field)
# and runtime-indexed `Array`/`Dict` containers completely untouched.
function emit_arg_path_group!(
    emit_leaf!::F,
    out_stmts::Vector{Expr},
    roots,
    entries::Vector{ArgPathEntry},
    split_counter::Base.RefValue{Int},
) where {F}
    isempty(entries) && return out_stmts
    if !_has_no_huge_inline_fields(roots)
        distinct_roots = Int[]
        root_groups = Dict{Int,Vector{ArgPathEntry}}()
        for e in entries
            r = e.prefix_steps[1]::Int
            grp = get!(root_groups, r) do
                push!(distinct_roots, r)
                return ArgPathEntry[]
            end
            push!(grp, e)
        end
        for r in distinct_roots
            grp = root_groups[r]
            child_obj = _safe_traced_getfield(roots, r)
            all_direct_leaves = all(e -> length(e.prefix_steps) == 1, grp)
            if all_direct_leaves
                _walk_arg_path_group!(
                    out_stmts,
                    child_obj,
                    :(getindex(_args_vec, _vidx)),
                    grp,
                    2,
                    split_counter,
                    emit_leaf!;
                    arg_root_idx=0,
                    from_any_vec=true,
                    parent_param=Pair{Symbol,Any}(:_vidx, r),
                )
            else
                _walk_arg_path_group!(
                    out_stmts,
                    child_obj,
                    :(getindex(_args_vec, $r)),
                    grp,
                    2,
                    split_counter,
                    emit_leaf!;
                    arg_root_idx=0,
                    from_any_vec=true,
                )
            end
        end
        return out_stmts
    end
    _walk_arg_path_group!(out_stmts, roots, :args, entries, 1, split_counter, emit_leaf!)
    return out_stmts
end

"""
    codegen_getfield_path(root, root_code, path, params, var_prefix)

Generate the `traced_getfield` chain accessing `path` inside `root` (whose AST is
`root_code`). Any runtime container indices along `path` (see [`is_runtime_index`](@ref))
are pushed to `params` so paths of the same structural shape emit identical ASTs.
"""
function codegen_getfield_path(
    @nospecialize(root),
    root_code,
    path,
    params::Vector{Pair{Symbol,Any}},
    var_prefix::Symbol,
)
    code = root_code
    field_codes = Any[]
    obj = root
    nidx = 0
    npath = length(path)
    for (k, p) in enumerate(path)
        if is_runtime_index(obj, p)
            nidx += 1
            var = Symbol(var_prefix, nidx)
            push!(params, var => p)
            field_code = var
        else
            field_code = Meta.quot(p)
        end
        push!(field_codes, field_code)
        code = :(traced_getfield($code, $field_code))
        if k < npath
            obj = _safe_traced_getfield(obj, p)
        end
    end
    return code, field_codes
end

"""
    codegen_flatten!

Generate Julia code to extract the XLA buffers from input arguments.
The name is due to its similarity to the `flatten` function in `jax.tree_util.register_pytree_node`.

Buffers are written into a `Vector` named `linearized_args` rather than `N` distinct local
variables, and each argument's extraction is emitted as a parameterized [`thunk_chunk`](@ref)
so arguments of the same structural shape share a single compiled function (see #3511).

# Arguments

- `linear_args`: A list of arguments to be flattened.
- `roots`: The concrete root arguments indexed by `args`.

# Returns

- `nbuffers`: Total number of buffers stored in `linearized_args`.
- `flatten_code`: A list of `Expr`s to extract the XLA buffers from the input arguments.
- `resharded_inputs`: Map from resharded input paths to `(device_sym, param_sharding_expr)`.

# Note

The _linearized arguments_ do not directly refer to the arguments that have been flattened into a single list.
"""
function codegen_flatten!(
    linear_args,
    seen_args,
    roots,
    is_sharded::Bool,
    linear_parameter_shardings,
    client,
    ndevices::Int,
    donated_args_mask::Vector{Bool},
)
    nbuffers = 0
    flatten_code = Expr[]
    runtime = XLA.runtime(client)
    resharded_inputs = Dict{Tuple,Any}()
    has_donated_args = any(donated_args_mask)

    inv_seen_args = if is_sharded
        result = Reactant.OrderedIdDict()
        for (k, v) in seen_args
            result[v] = k
        end
        result
    else
        nothing
    end

    needs_param_shardings = false
    nargs = length(linear_args)
    entries = Vector{ArgPathEntry}(undef, nargs)

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
        entries[i] = ArgPathEntry(path[2:end], (), (i, arg, path))
    end

    emit_arg_path_group!(
        flatten_code, roots, entries, Ref(0)
    ) do out_stmts, flatcode, _suffix_codes, path_params, (i, arg, path)
        params = Pair{Symbol,Any}[:arg_idx => i]
        has_donated_args && push!(params, :arg_path => path)
        append!(params, path_params)
        chunk = Expr[:(carg = $flatcode)]

        if runtime isa Val{:PJRT}
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

                    push!(chunk, :(usbuf = carg.data))
                    @assert length(carg.sharding.mesh) == ndevices
                    for j in 1:ndevices
                        nbuffers += 1
                        offset = (j - 1) * nargs
                        slot_expr = offset == 0 ? :arg_idx : :($offset + arg_idx)
                        push!(
                            chunk,
                            :(@inbounds linearized_args[$slot_expr] = XLA.synced_buffer(
                                getindex(usbuf, $(j))
                            )),
                        )
                    end
                else
                    if DEBUG_DISABLE_RESHARDING[]
                        error("Resharding is disabled. Problematic input:\ntypeof: \
                               $(typeof(carg))\nsize: $(size(carg))\nsharding: \
                               $(carg.sharding)\nInput Index: $(i)\nInput Path: \
                               $(path[3:end])")
                    end

                    push!(chunk, :(usbuf = carg))
                    device_sym = Symbol(:input_device_, i)
                    push!(chunk, :($device_sym = Reactant.XLA.device(usbuf)))
                    needs_param_shardings = true
                    resharded_inputs[path[3:end]] = (device_sym, :xla_parameter_sharding)

                    device_to_array_slices, _ = XLA.sharding_to_concrete_array_indices(
                        condensed_op_sharding, size(carg), 0:(ndevices - 1)
                    )

                    # Extract the buffer_slice
                    buf_slice = Dict{eltype(device_to_array_slices),Symbol}()
                    counter = 0
                    for j in 1:ndevices
                        sliced_buf = Symbol(:sliced_buf_, counter)
                        slice = device_to_array_slices[j]
                        haskey(buf_slice, slice) && continue
                        counter += 1
                        push!(
                            chunk,
                            :(
                                $sliced_buf = only(
                                    Reactant._fast_slice(usbuf, $(slice...)).data
                                )
                            ),
                        )
                        buf_slice[slice] = sliced_buf
                    end

                    for j in 1:ndevices
                        slice = device_to_array_slices[j]
                        nbuffers += 1
                        offset = (j - 1) * nargs
                        slot_expr = offset == 0 ? :arg_idx : :($offset + arg_idx)
                        push!(
                            chunk,
                            :(@inbounds linearized_args[$slot_expr] = XLA.copy_buffer_to_device(
                                XLA.synced_buffer($(buf_slice[slice])),
                                XLA.get_device(thunk.client, global_mesh.device_ids[$(j)]),
                            )),
                        )
                    end
                end
            else
                push!(chunk, :(usbuf = carg.data))
                nbuffers += 1
                if arg isa TracedRArray || arg isa TracedRNumber
                    push!(
                        chunk,
                        :(@inbounds linearized_args[arg_idx] = only(
                            XLA.synced_buffer(usbuf)
                        )),
                    )
                else
                    error("Unsupported type $(typeof(arg))")
                end
            end
        elseif runtime isa Val{:IFRT}
            push!(chunk, :(usbuf = carg.data))
            nbuffers += 1
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

                    push!(
                        chunk,
                        :(@inbounds linearized_args[arg_idx] = XLA.synced_buffer(usbuf)),
                    )
                else
                    if DEBUG_DISABLE_RESHARDING[]
                        error("Resharding is disabled. Problematic input:\ntypeof: \
                               $(typeof(carg))\nsize: $(size(carg))\nsharding: \
                               $(carg.sharding)\nInput Index: $(i)\nInput Path: \
                               $(path[3:end])")
                    end

                    device_sym = Symbol(:input_device_, i)
                    push!(chunk, :($device_sym = Reactant.XLA.device(usbuf)))
                    needs_param_shardings = true
                    resharded_inputs[path] = (device_sym, :(xla_parameter_sharding[$(i)]))

                    push!(
                        chunk,
                        :(@inbounds linearized_args[arg_idx] = ifrt_resharded_buffer(
                            $(Reactant.unwrapped_eltype(carg)),
                            usbuf,
                            $(size(carg)),
                            thunk.client,
                            carg.sharding,
                            thunk.global_device_ids,
                            xla_parameter_sharding[arg_idx],
                        )),
                    )
                end
            else
                push!(
                    chunk, :(@inbounds linearized_args[arg_idx] = XLA.synced_buffer(usbuf))
                )
            end
        else
            error("Unsupported runtime $runtime")
        end

        if has_donated_args
            # Important to mark donated after we have extracted the data
            push!(
                chunk,
                :(donate_argument!(
                    donated_args_mask, carg, arg_idx, donated_buffers, arg_path
                )),
            )
        end
        push!(out_stmts, thunk_chunk(chunk...; params, reorderable=true))
    end

    if needs_param_shardings
        pushfirst!(
            flatten_code,
            :(xla_parameter_sharding = Reactant.XLA.get_parameter_shardings(thunk.exec)),
        )
    end
    buffer_type = runtime isa Val{:PJRT} ? XLA.PJRT.Buffer : XLA.IFRT.Array
    pushfirst!(flatten_code, :(linearized_args = Vector{$buffer_type}(undef, $nbuffers)))

    return nbuffers, flatten_code, resharded_inputs
end

function donate_argument!(
    donated_args_mask,
    carg::Union{ConcretePJRTNumber,ConcretePJRTArray},
    i::Int,
    donated_buffers,
    path,
)
    if donated_args_mask[i]
        if donated_buffers !== nothing
            buffers = map(Base.Fix2(getfield, :buffer), carg.data)
            if buffers in donated_buffers
                error(
                    "Donated buffer $(carg.data) is already marked as donated. Can't donate \
                     the same buffer multiple times. The argument is present at $(path)",
                )
            end
            push!(donated_buffers, buffers)
        end
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
        if donated_buffers !== nothing
            if carg.data.buffer in donated_buffers
                error(
                    "Donated buffer $(carg.data) is already marked as donated. Can't donate \
                     the same buffer multiple times. The argument is present at $(path)",
                )
            end
            push!(donated_buffers, carg.data.buffer)
        end
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
    roots,
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
    unresharded_arrays_cache = Dict{Any,Symbol}()
    unresharded_code = Expr[]
    unflatten_code = Expr[]
    used_shardinfo = Set{Any}()

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

    split_counter = Ref(0)
    resarg_entries = ArgPathEntry[]
    track_alias_paths = DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]

    for (i, (concrete_res_name, result, shard_info)) in
        enumerate(zip(concretized_res_names, linear_results, linear_result_shard_info))
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == resprefix || p[1] == resargprefix)
            )...,
        )
        for path in paths
            if path[1] == resprefix
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

                need_to_unreshard = get(resharded_inputs, (:args, path[2:end]...), nothing)
                if need_to_unreshard !== nothing
                    @assert runtime isa Val{:IFRT} "PJRT is not supported here. Use IFRT \
                                                    instead."
                end

                root_idx = path[2]
                subpath = path[3:end]

                concrete_res_expr = if need_to_unreshard !== nothing
                    if !haskey(unresharded_arrays_cache, concrete_res_name)
                        unreshard_sym = Symbol(
                            :unresharded_buffer_, length(unresharded_arrays_cache) + 1
                        )
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
                    unresharded_arrays_cache[concrete_res_name]
                else
                    i
                end

                if isempty(subpath) ||
                    _is_statically_concrete_target(typeof(roots), (root_idx, subpath...))
                    push!(
                        resarg_entries,
                        ArgPathEntry(
                            (root_idx, subpath...),
                            (),
                            (:direct, concrete_res_expr, subpath),
                        ),
                    )
                elseif length(subpath) > 1
                    needs_cache_dict = true
                    push!(
                        resarg_entries,
                        ArgPathEntry(
                            (root_idx, subpath[1:(end - 2)]...),
                            (subpath[end - 1], subpath[end]),
                            (:at_parent, concrete_res_expr, subpath),
                        ),
                    )
                else
                    needs_cache_dict = true
                    push!(
                        resarg_entries,
                        ArgPathEntry(
                            (root_idx, subpath[1:(end - 1)]...),
                            (subpath[end],),
                            (:at_field, concrete_res_expr, subpath),
                        ),
                    )
                end
            end
        end
    end

    if needs_cache_dict
        push!(
            unflatten_code,
            :(
                $cache_dict = IdDict{
                    Union{TracedRArray,TracedRNumber,Reactant.TracedEnum},
                    Union{$ctypes,Reactant.ConcreteEnum},
                }()
            ),
        )
    end

    emit_arg_path_group!(
        unflatten_code, roots, resarg_entries, split_counter
    ) do out_stmts,
    target_expr, suffix_codes, path_params,
    (mode, concrete_res_expr, subpath)
        params = Pair{Symbol,Any}[]
        res_expr = if concrete_res_expr isa Int
            push!(params, :res_idx => concrete_res_expr)
            :(@inbounds linearized_results[res_idx])
        else
            concrete_res_expr
        end
        res_path_expr = if track_alias_paths
            push!(params, :res_path => subpath)
            :res_path
        else
            :()
        end
        append!(params, path_params)
        stmt = if mode === :at_parent
            :(traced_setfield_buffer_at_parent!(
                $(runtime),
                $(cache_dict),
                $(res_expr),
                $(target_expr),
                $(suffix_codes[1]),
                $(suffix_codes[2]),
                $(res_path_expr),
            ))
        elseif mode === :at_field
            # TODO(#2233): we might need to handle sharding here
            :(traced_setfield_buffer!(
                $(runtime),
                $(cache_dict),
                $(res_expr),
                $(target_expr),
                $(suffix_codes[1]),
                $(res_path_expr),
            ))
        else
            :(traced_setfield!($(target_expr), :data, $(res_expr), $(res_path_expr)))
        end
        push!(out_stmts, thunk_chunk(stmt; params, reorderable=true))
    end

    result_cache = IdDict{Any,Symbol}()
    var_idx = Ref(0)
    resultgen_code = Expr[]
    preserved_result_code = Expr[]
    pres_result_items = Tuple{Tuple,Tuple}[]

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

            push!(pres_result_items, (path, argpath))
        end
    end

    if length(pres_result_items) <= 4 &&
        _has_no_huge_inline_fields(roots) &&
        all(
            ((_, argpath),) -> _is_small_typeassert(Core.Typeof(roots[argpath[2]])),
            pres_result_items,
        )
        for (path, argpath) in pres_result_items
            in(path, keys(result_stores)) && continue
            argres = :(args[$(argpath[2])])
            for p in argpath[3:end]
                argres = :(traced_getfield($argres, $(Meta.quot(p))))
            end
            result_stores[path] = :($argres.data)
        end
    else
        pres_entries = ArgPathEntry[]
        for (path, argpath) in pres_result_items
            in(path, keys(result_stores)) && continue
            slot = length(pres_entries) + 1
            push!(pres_entries, ArgPathEntry(argpath[2:end], (), slot))
            result_stores[path] = :(preserved_result_buffers[$slot])
        end
        n_pres = length(pres_entries)
        if n_pres > 0
            pres_buf_type = if runtime isa Val{:IFRT}
                XLA.IFRT.AsyncArray
            elseif path_to_shard_info === nothing
                Tuple{XLA.PJRT.AsyncBuffer}
            else
                Any
            end
            push!(
                preserved_result_code,
                :(preserved_result_buffers = Vector{$pres_buf_type}(undef, $n_pres)),
            )
            emit_arg_path_group!(
                preserved_result_code, roots, pres_entries, split_counter
            ) do out_stmts, target_expr, _suffix_codes, path_params, slot
                params = Pair{Symbol,Any}[:_pres_slot => slot, path_params...]
                push!(
                    out_stmts,
                    thunk_chunk(
                        :(@inbounds preserved_result_buffers[_pres_slot] =
                            ($target_expr).data);
                        params,
                        reorderable=true,
                    ),
                )
            end
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

    # If some argument is mutated, change them to point to the correct concrete results.
    #
    # The source-buffer reads are split from the destination writes so each can be placed
    # at the correct point relative to the execution-result write-back generated in
    # `unflatten_code`:
    #   * A result that aliases an input argument (it has an `:args` path) takes the
    #     source argument's *pre-execution* buffer, so that read must be captured before
    #     any of the result assignments (`resultgen_code`, the `result` construction, and
    #     the write-back) read from or overwrite the arguments. Those reads go at the very
    #     front of the returned body.
    #   * A pure `:resargs` mutation takes the *post-execution* value, so that read
    #     happens after the write-back.
    # In all cases the writes are applied after the write-back so an aliased argument is
    # not clobbered by it.
    orig_buffer_available = trues(length(linear_args))
    pre_execution_read_code = Expr[]
    post_execution_read_code = Expr[]
    alias_write_code = Expr[]
    n_aliased = codegen_unflatten_aliased_args!(
        pre_execution_read_code,
        post_execution_read_code,
        alias_write_code,
        preserved_args,
        linear_args,
        roots,
        resharded_inputs,
        orig_buffer_available,
        runtime,
        split_counter,
    )

    # generate return object which stores the concrete results in some arbitrary way
    body_code = Expr[
        pre_execution_read_code...,
        preserved_result_code...,
        map(thunk_chunk, unresharded_code)...,
        map(thunk_chunk, resultgen_code)...,
        thunk_chunk(:(result = $result_code)),
        unflatten_code...,
        post_execution_read_code...,
        alias_write_code...,
    ]

    if n_aliased > 0
        alias_buf_type = if runtime isa Val{:IFRT}
            XLA.IFRT.AsyncArray
        elseif path_to_shard_info === nothing
            Tuple{XLA.PJRT.AsyncBuffer}
        else
            Any
        end
        pushfirst!(
            body_code, :(aliased_arg_buffers = Vector{$alias_buf_type}(undef, $n_aliased))
        )
    end

    if DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]
        push!(body_code, :(empty!(DEBUG_BUFFER_POINTERS_STORE_DICT)))
    end

    return (body_code, used_shardinfo)
end

function codegen_unflatten_aliased_args!(
    pre_execution_read_code,
    post_execution_read_code,
    write_code,
    preserved_args,
    linear_args,
    roots,
    resharded_inputs,
    orig_buffer_available,
    runtime,
    split_counter::Base.RefValue{Int},
)
    n_aliased = 0
    pre_read_entries = ArgPathEntry[]
    post_read_entries = ArgPathEntry[]
    write_entries = ArgPathEntry[]
    track_alias_paths = DEBUG_ALIASED_BUFFER_ASSIGNMENT_ERROR[]

    for (result, arg_idx) in preserved_args
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :args || p[1] == :resargs)
            )...,
        )
        isempty(paths) && continue

        arg = linear_args[arg_idx + 1]
        argpath = only((
            p for p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
        ))
        alias_slot = 0

        if any(parg -> parg[2] == arg_idx, preserved_args)
            orig_buffer_available[arg_idx + 1] = false
        end

        # A result that also lives at an `:args` path aliases an input argument, so its
        # source buffer must be captured before the execution results overwrite the
        # arguments. Otherwise it is a pure mutation and the post-execution value is used.
        aliases_input = any(p -> p[1] == :args, paths)
        read_entries = aliases_input ? pre_read_entries : post_read_entries

        seen_dsts = Set{Any}()
        for path in paths
            initial_path = path
            # We can optimize cases where we set the arg to itself
            if path[2:end] == argpath[2:end]
                continue
            end
            # The destination position is independent of the path prefix; an `:args` and a
            # `:resargs` path can resolve to the same argument slot. Write it only once.
            path[2:end] in seen_dsts && continue
            push!(seen_dsts, path[2:end])

            need_to_unreshard = get(resharded_inputs, (:args, argpath[2:end]...), nothing)
            if need_to_unreshard !== nothing
                # TODO(@avik-pal): I need an MWE to debug this codepath
                error("TODO(#2234): Not yet Implemented. Open an issue on Reactant.jl.")
            end

            if alias_slot == 0
                # this traced array take an identity value derived from another argument but
                # is its own traced array. As such, it needs to allocate a new buffer instead of using the arg directly.
                needs_copy =
                    initial_path[1] === :args &&
                    argpath[1] === :args &&
                    !orig_buffer_available[arg_idx + 1]
                orig_buffer_available[arg_idx + 1] = false

                n_aliased += 1
                alias_slot = n_aliased
                push!(
                    read_entries, ArgPathEntry(argpath[2:end], (), (alias_slot, needs_copy))
                )
            end

            push!(write_entries, ArgPathEntry(path[2:end], (), (alias_slot, path[3:end])))
        end
    end

    for (entries_vec, out_code) in (
        (pre_read_entries, pre_execution_read_code),
        (post_read_entries, post_execution_read_code),
    )
        emit_arg_path_group!(
            out_code, roots, entries_vec, split_counter
        ) do out_stmts, argres, _suffix_codes, path_params, (alias_slot, needs_copy)
            read_params = Pair{Symbol,Any}[:_alias_slot => alias_slot, path_params...]
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
            # Capture the source buffer into `aliased_arg_buffers` so the actual write
            # can be applied later without keeping `O(N)` distinct locals live.
            push!(
                out_stmts,
                thunk_chunk(
                    :(@inbounds aliased_arg_buffers[_alias_slot] = $argres_data);
                    params=read_params,
                    reorderable=true,
                ),
            )
        end
    end

    emit_arg_path_group!(
        write_code, roots, write_entries, split_counter
    ) do out_stmts, res, _suffix_codes, path_params, (alias_slot, dst_subpath)
        write_params = Pair{Symbol,Any}[:_alias_slot => alias_slot]
        dst_path_expr = if track_alias_paths
            push!(write_params, :_dst_path => dst_subpath)
            :_dst_path
        else
            :()
        end
        append!(write_params, path_params)
        push!(
            out_stmts,
            thunk_chunk(
                :(traced_setfield!(
                    $res, :data, @inbounds(aliased_arg_buffers[_alias_slot]), $dst_path_expr
                ));
                params=write_params,
                reorderable=true,
            ),
        )
    end

    return n_aliased
end

"""
    codegen_xla_call

Generate Julia code to call the XLA executable.

# Arguments

- `nresults`: The number of results to expect.
- `is_sharded`: Whether the function is sharded.
- `ndevices`: The number of devices.
- `is_pure`: Whether the function being compiled is pure (i.e., has no side effects)
"""
function codegen_xla_call(nresults, is_sharded::Bool, ndevices::Int, is_pure::Bool)
    concretized_res_names = Any[:(linearized_results[$i]) for i in 1:nresults]

    xla_call_code = if nresults == 0 && is_pure
        :()
    elseif is_sharded
        :(
            linearized_results = XLA.execute(
                thunk.exec,
                linearized_args,
                donated_args_mask,
                $nresults,
                Val($ndevices),
            )
        )
    else
        :(
            linearized_results = XLA.execute_sharded(
                thunk.exec, thunk.device, linearized_args, donated_args_mask, $nresults
            )
        )
    end

    return concretized_res_names, xla_call_code
end

# generate the shard info code. we cannot embed any mesh information into the code here,
# else serialization will be incorrect.
function codegen_shard_info(
    is_sharded, nresults::Int, linear_results, output_reactant_shardings, exec, ndevices
)
    !is_sharded && return Expr[], Any[nothing for _ in 1:nresults]

    shard_info_code = Expr[]
    output_hlo_shardings_var = missing
    output_hlo_shardings = XLA.get_output_shardings(exec)

    linear_result_shard_info = Vector{Any}(undef, length(linear_results))
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

            linear_result_shard_info[i] = :(Sharding.ShardInfo(
                Sharding.HloSharding(
                    convert(XLA.HloSharding, $(output_hlo_shardings_var)[$(i)]),
                    global_mesh,
                    $(ntuple(Returns(true), length(res_size))),
                    $(ntuple(Returns(-1), length(res_size))),
                    nothing,
                ),
                $(array_slices),
            ))
        else
            mesh = output_reactant_shardings[i].mesh
            mesh_key = (mesh.logical_device_ids, mesh.axis_names, mesh.axis_sizes)
            if haskey(mesh_codegen_cache, mesh_key)
                mesh_name = mesh_codegen_cache[mesh_key]
            else
                mesh_name = Symbol(:mesh_, length(mesh_codegen_cache) + 1)
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
            linear_result_shard_info[i] =
                :(Sharding.ShardInfo($(new_sharding), $(array_slices)))
        end
    end
    return shard_info_code, linear_result_shard_info
end
