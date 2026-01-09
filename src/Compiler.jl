module Compiler

using Reactant_jll
using Libdl: dlsym
using LinearAlgebra: BlasInt
using Functors: Functors

import ..Reactant:
    Reactant,
    MLIR,
    XLA,
    Sharding,
    ConcretePJRTArray,
    ConcretePJRTNumber,
    ConcreteIFRTArray,
    ConcreteIFRTNumber,
    TracedRArray,
    TracedRNumber,
    RArray,
    OrderedIdDict,
    make_tracer,
    TracedToConcrete,
    append_path,
    ancestor,
    TracedType
import Reactant: OptimizeCommunicationOptions, ShardyPropagationOptions, CompileOptions
using Reactant_jll: Reactant_jll

import ..ReactantCore: correct_maybe_bcast_call

const DEBUG_PRINT_CODEGEN = Ref(false)
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
        obj::Union{Reactant.AbstractConcreteArray,Reactant.AbstractConcreteNumber}
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
    @nospecialize(obj::Reactant.AbstractConcreteNumber), field, val, path
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

@inline traced_setfield!(@nospecialize(obj), field, val, path) =
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

@inline function traced_setfield!(@nospecialize(obj::Dict), field, val, path)
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

function traced_setfield_buffer!(::Val, cache_dict, val, concrete_res, obj, field, path)
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

function create_result(
    tocopy::T,
    path,
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
) where {T}
    if !isstructtype(typeof(tocopy))
        error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
    end

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

function create_result(
    tocopy::ConcretePJRTNumber{T,D},
    path,
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
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
                error("TODO: Not yet Implemented. Use IFRT for this.")
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
    path,
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
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
                error("TODO: Not yet Implemented.")
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
    path,
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
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
                error("TODO: Not yet Implemented. Use IFRT for this.")
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
    path,
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
    path,
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
    path,
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
    path,
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
    for (i, (k, v)) in enumerate(pairs(tocopy))
        push!(elems, create_result(v, append_path(path, i), args...))
    end
    return :(NamedTuple{$K}(($(elems...),)))
end

function create_result(
    tocopy::D,
    path,
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
            push!(
                resultgen_code,
                quote
                    @inbounds $sym[$k] = $subexpr
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
    path,
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
    return Meta.quot(:($(tocopy)))
end

function create_result(
    tocopy::Union{Integer,AbstractFloat,AbstractString,Nothing,Type,Symbol,Char},
    path,
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
    return Meta.quot(tocopy)
end

const WHILE_CONCAT = Ref(false)
const DUS_TO_CONCAT = Ref(false)
const SUM_TO_REDUCEWINDOW = Ref(false)
const SUM_TO_CONV = Ref(false)
const AGGRESSIVE_SUM_TO_CONV = Ref(false)
const AGGRESSIVE_PROPAGATION = Ref(false)
const DUS_SLICE_SIMPLIFY = Ref(true)
const CONCATS_TO_DUS = Ref(false)
const WHILE_UNROLL_THRESHOLD = Ref(4)

# Optimization passes via transform dialect
function optimization_passes(
    compile_options::CompileOptions;
    sroa::Bool=false,
    dus_to_concat::Bool=false,
    recognize_comms::Bool=true,
    lower_comms::Bool=true,
    backend::String="gpu",
    is_sharded::Bool=false,
    raise_shlo_to_blas_lapack::Bool=true,
)
    (; max_constant_threshold) = compile_options

    transform_passes_list = [
        "patterns=compare_op_canon<16>",
        "transpose_transpose<16>",
        "broadcast_in_dim_op_canon<16>",
        "convert_op_canon<16>",
        "dynamic_broadcast_in_dim_op_not_actually_dynamic<16>",
        "chained_dynamic_broadcast_in_dim_canonicalization<16>",
        "dynamic_broadcast_in_dim_all_dims_non_expanding<16>",
        "noop_reduce_op_canon<16>",
        "empty_reduce_op_canon<16>",
        "dynamic_reshape_op_canon<16>",
        "get_tuple_element_op_canon<16>",
        "real_op_canon<16>",
        "imag_op_canon<16>",
        "conj_complex_negate<16>",
        "get_dimension_size_op_canon<16>",
        "reshape_op_canon<16>",
        "merge_consecutive_reshapes<16>",
        "transpose_is_reshape<16>",
        "zero_extent_tensor_canon<16>",
        "cse_broadcast_in_dim<16>",
        "cse_slice<16>",
        "cse_transpose<16>",
        "cse_convert<16>",
        "cse_dot_general<16>",
        "cse_reshape<16>",
        "cse_mul<16>",
        "cse_div<16>",
        "cse_add<16>",
        "cse_subtract<16>",
        "cse_min<16>",
        "cse_max<16>",
        "cse_neg<16>",
        "cse_abs<16>",
        "cse_concatenate<16>",
        "cse_compare<16>",
        "cse_select<16>",
        "concatenate_op_canon<16>($max_constant_threshold)",
        "select_op_canon<16>($max_constant_threshold)",
        "add_simplify<16>",
        "sub_simplify<16>",
        "and_simplify<16>",
        "max_simplify<16>",
        "min_simplify<16>",
        "or_simplify<16>",
        "xor_simplify<16>",
        "mul_simplify<16>",
        "div_simplify<16>",
        "rem_simplify<16>",
        "pow_simplify<16>",
        "simplify_extend<16>",
        "simplify_wrap<16>",
        "simplify_rotate<16>",
        "extend_splat<16>",
        "noop_slice<16>",
        "noop_reverse<16>",
        "slice_slice<16>",
        "dynamic_slice_slice<16>",
        "slice_dynamic_slice<16>",
        "dynamic_slice_dynamic_slice<16>",
        "shift_right_logical_simplify<16>",
        "slice_simplify<16>",
        "convert_simplify<16>",
        "dynamic_slice_to_static<16>",
        "dynamic_update_slice_elim<16>",
        "concat_to_broadcast<16>",
        "reduce_to_reshape<16>",
        "broadcast_to_reshape<16>",
        "slice_internal",
        "iota_simplify<16>($max_constant_threshold)",
        "broadcast_in_dim_simplify<16>($max_constant_threshold)",
        "convert_concat<1>",
        "dynamic_update_to_concat<1>",
        "slice_of_dynamic_update<1>",
        "slice_elementwise<1>",
        "dot_reshape_dot<1>",
        "concat_fuse<1>",
        "concat_push_binop_add<1>",
        "concat_push_binop_mul<1>",
        "reduce_concat<1>",
        "slice_concat<1>",
        "concat_slice<1>",
        "select_op_used_within_if<1>",
        "bin_broadcast_splat_add<1>",
        "bin_broadcast_splat_subtract<1>",
        "bin_broadcast_splat_div<1>",
        "bin_broadcast_splat_mul<1>",
        "dot_general_simplify<16>",
        "transpose_simplify<16>",
        "reshape_empty_broadcast<1>",
        "broadcast_reshape<1>",
        "transpose_dot_reorder<1>",
        "dot_transpose<1>",
        "transpose_convolution<1>",
        "convolution_transpose<1>",
        "convert_convert_float<1>",
        "convert_convert_int<1>",
        "reshape_iota<1>",
        "broadcast_reduce<1>",
        "slice_dot_general<1>",
        "if_inline<1>",
        "if_to_select<1>",
        "divide_sqrt_to_multiply_rsqrt<16>",
        "associative_binary_op_reordering<1>",
        "transpose_broadcast_in_dim_to_broadcast_in_dim<16>",
        "replace_neg_add_with_subtract",
        "replace_subtract_neg_with_add",
        "binop_const_simplify",
        "not_select_simplify",
        "common_compare_expression_rewrite",
        "compare_select_simplify",
        "while_simplify<1>(1)",
        "if_remove_unused",
        "transpose_reshape_to_broadcast",
        "reshape_transpose_to_broadcast",
        "reshape_broadcast",
        "dus_dus",
        "dus_dus_concat",
        "abs_positive_simplify",
        "transpose_elementwise_transpose",
        "select_comp_iota_const_simplify<1>",
        "sign_abs_simplify<1>",
        "broadcastindim_is_reshape",
        "reduce_window_wrap<1>",
        "slice_reduce_window<1>",
        "while_deadresult",
        "while_dus",
        "while_updatewithoutcorners",
        "while_op_induction_replacement",
        "dus_concat",
        "dusdus_to_duspad",
        "slice_dus_to_concat",
        "sink_dus",
        "hoist_slice",
        "while_induction_reduction",
        "slice_broadcast",
        "associative_common_mul_op_reordering",
        "slice_select_to_select_slice",
        "slice_if",
        "dus_to_i32",
        "slice_extend",
        "slice_of_updatewithoutcorners",
        "concat_wrap",
        "cse_updatewithoutcorners<16>",
        "cse_extend<16>",
        "cse_wrap<16>",
        "cse_rotate<16>",
        "concat_concat_axis_swap",
        "concat_concat_to_dus",
        "broadcast_iota_simplify",
        "select_comp_iota_to_dus",
        "compare_cleanup",
        "broadcast_compare",
        "not_compare",
        "broadcast_iota",
        "cse_iota",
        "compare_iota_const_simplify",
        "reshuffle_ands_compares",
        "square_abs_simplify",
        "divide_divide_simplify",
        "concat_reshape_slice",
        "full_reduce_reshape_or_transpose",
        "concat_reshape_reduce",
        "concat_elementwise",
        "reduce_reduce",
        "conj_real",
        "select_broadcast_in_dim",
        "if_op_lift_common_ops",
        "involution_neg_simplify",
        "involution_conj_simplify",
        "involution_not_simplify",
        "real_conj_simplify",
        "conj_complex_simplify",
        "split_convolution_into_reverse_convolution",
        # TODO we want to enable but may cause an infinite compile time
        # "concat_to_onedim_dusslice",
        # TODO expose an option to enable this
        # "chained_multiply_to_power",
        "power_multiply_to_power",
        "log_simplify",
        "neg_mul_const_simplify",
        "neg_div_const_simplify",
        "reshape_deletions_broadcast_in_dim_simplify",
        "reshape_insertions_broadcast_in_dim_simplify",
        "dot_general_reshape",
        "widen_wrap",
        "widen_extend",
        "elementwise_pad",
        "compare_negate_const_simplify",
        "select_simplify",
        "concatenate_subtract_to_subtract_pad",
        "concatenate_broadcast_in_dim",
        "compare_abs",
        # "compare_mul",
        "compare_convert",
        "add_selects",
        "self_subtract_to_convolution_like(0)",
        "self_add_to_convolution_like(0)",
        "self_mul_to_convolution_like(0)",
        "subtract_multiply_const_to_add_mul_const",
        "trivial_reduce_window_to_reduce_op",
        "case_to_if",
        "dot_general_add_distributive_simplify",
        "dot_general_subtract_distributive_simplify",
        "remove_no_ops_from_while_loop",
        "while_is_copy_simplify",
        "split_variadic_scatter_op",
        "dynamic_slice_simplify",
        "enzyme_hlo_unroll($(WHILE_UNROLL_THRESHOLD[]))",
        "divide_negated_operands_simplify",
        "multiply_negated_operands_simplify",
        "factor_scalars_in_dot_general",
        "reduce_mul_to_dot_general",
        "dot_general_broadcast_in_dim",
        "dot_general_broadcast_in_dim_sort_dims",
        "dus_dynamic_slice_simplify",
        "while_dus_dus_simplify",
        "while_dus_ds_simplify",
        "reshape_slice_reshape",
        "dynamic_slice_elementwise",
        "dot_general_remove_batch_dimensions",
        "delete_dims_reduce",
        "reduce_delete_dims",
        "dot_general_insert_dim_contraction_simplification",
        "fuse_reshape_collapse_or_expand_dims_into_reduce",
        "split_reduce_add_mul_to_add_dot_general",
        "recognize_from_constant($(max_constant_threshold))",
    ]

    if !is_sharded
        # these passes don't have optimized sharding implementations
        if raise_shlo_to_blas_lapack
            if !compile_options.disable_structured_tensors_detection_passes
                append!(transform_passes_list, ["dot_general_to_syrk"])
            end
        end
    end

    if !compile_options.disable_structured_tensors_passes
        append!(
            transform_passes_list,
            [
                "transpose_syrk_to_syrk",
                "fuse_mul_into_syrk",
                "fuse_add_into_syrk",
                "dot_general_only_diagonal_access",
                "transpose_symmetric_simplify",
                "syrk_simplify_output_uplo",
            ],
        )
    end

    if !compile_options.disable_scatter_gather_optimization_passes
        append!(
            transform_passes_list,
            [
                # scatter patterns
                "scatter_to_dynamic_update_slice<1>",
                "scatter_multiply_simplify",
                "scatter_sub_simplify",
                "scatter_add_simplify",
                "scatter_div_simplify",
                "unary_elementwise_scatter_simplify",
                "scatter_indices_are_unique",
                ## const prop patterns
                "scatter_update_computation_const_prop",
                # gather patterns
                "dynamic_gather_op_is_not_dynamic<16>",
                "gather_op_canon<16>",
                "gather_elementwise",
                "gather_of_scatter_simplify",
                ## const prop patterns
                "gather_const_prop",
                "scatter_const_fold($max_constant_threshold)",
                "cse_gather",
                "cse_scatter",
            ],
        )
    end

    if (
        !compile_options.disable_scatter_gather_optimization_passes &&
        !compile_options.disable_structured_tensors_passes
    )
        append!(transform_passes_list, ["diagonal_tensor_dot_general_rewrite"])
    end

    if !compile_options.disable_slice_to_batch_passes
        append!(
            transform_passes_list,
            [
                "dot_general_slice_to_batch",
                "gather_slice_to_batch",
                "iota_slice_to_batch",
                "reduce_slice_to_batch",
                "sort_slice_to_batch",
                "transpose_slice_to_batch",
                "broadcastindim_slice_to_batch",
                "reducewindow_slice_to_batch",
                "elementwise_slice_to_batch",
                "convolution_slice_to_batch",
            ],
        )
    end

    if !compile_options.disable_reduce_slice_fusion_passes
        append!(
            transform_passes_list,
            [
                "add_reduce_slice_fusion",
                "mul_reduce_slice_fusion",
                "min_reduce_slice_fusion",
                "max_reduce_slice_fusion",
                "and_reduce_slice_fusion",
                "xor_reduce_slice_fusion",
                "or_reduce_slice_fusion",
            ],
        )
    end

    if !compile_options.disable_concat_to_batch_passes
        append!(
            transform_passes_list,
            [
                "concat_insert_dim_dot_general",
                "concat_insert_dim_gather",
                "concat_insert_dim_iota",
                "concat_insert_dim_reduce",
                "concat_insert_dim_sort",
                "concat_insert_dim_reduce_window",
                "concat_insert_dim_elementwise",
                "concat_insert_dim_convolution",
            ],
        )
    end

    if !compile_options.disable_loop_raising_passes
        append!(
            transform_passes_list,
            [
                "greedy_while_loop_batch_fission",
                "while_elementwise_reduction_to_reduce",
                "remove_loop_carried_dependencies_from_while_load_operations",
            ],
        )
    end

    if !compile_options.disable_licm_optimization_passes
        append!(
            transform_passes_list,
            [
                "dus_licm(0)",
                "slice_licm(0)",
                "elementwise_licm(0)",
                "concatenate_licm(0)",
                "while_licm<1>(1)",
                "transpose_licm(0)",
                "broadcastindim_licm(0)",
                "reshape_licm(0)",
                "dot_general_licm(0)",
                "reduce_licm(0)",
                "reduce_window_licm(0)",
                "reverse_licm(0)",
                "convolution_licm(0)",
                "dynamic_slice_licm(0)",
                "scatter_licm(0)",
                "gather_licm(0)",
                "iota_licm(0)",
                "rotate_licm(0)",
                "wrap_licm(0)",
                "extend_licm(0)",
            ],
        )
    end

    if !compile_options.disable_pad_optimization_passes
        append!(
            transform_passes_list,
            [
                "extend_pad",
                "dus_pad",
                "cse_pad<16>",
                "pad_simplify<16>($max_constant_threshold)",
                "select_pad_to_dus<1>",
                "and_pad_pad<1>",
                "negative_pad_to_slice<16>",
                "slice_pad<1>",
                "pad_reshape_pad<1>",
                "pad_pad<1>",
                "add_pad_pad_to_concat<1>",
                "concat_pad<1>",
                "reduce_pad<1>",
                "broadcast_pad<1>",
                "zero_product_reshape_pad<1>",
                "mul_zero_pad<1>",
                "div_zero_pad<1>",
                "binop_const_reshape_pad<1>",
                "binop_const_pad_add<1>",
                "binop_const_pad_subtract<1>",
                "binop_const_pad_mul<1>",
                "binop_const_pad_div<1>",
                "binop_binop_pad_pad_add<1>",
                "binop_binop_pad_pad_mul<1>",
                "binop_pad_pad_add<1>",
                "binop_pad_pad_subtract<1>",
                "binop_pad_pad_mul<1>",
                "binop_pad_pad_div<1>",
                "binop_pad_pad_min<1>",
                "binop_pad_pad_max<1>",
                "unary_pad_push_convert<1>",
                "unary_pad_push_tanh<1>",
                "unary_pad_push_exp<1>",
                "concat_to_pad<1>",
                "while_pad_induction_reduction",
                "pad_concat_to_concat_pad",
                "rotate_pad",
                "concat_multipad",
                "speculate_if_pad_to_select",
                "dus_to_dynamic_pad",
                "dynamic_pad_to_pad",
            ],
        )

        if !compile_options.disable_licm_optimization_passes
            push!(transform_passes_list, "pad_licm(0)")
        end
    end

    # constant prop patterns
    append!(
        transform_passes_list,
        [
            # unary constant propagation
            "chlo_inf_const_prop<16>",
            "gamma_const_prop<16>",
            "abs_const_prop<16>",
            "log_const_prop<1>",
            "log_plus_one_const_prop<1>",
            "is_finite_const_prop",
            "not_const_prop",
            "neg_const_prop",
            "sqrt_const_prop",
            "rsqrt_const_prop",
            "cos_const_prop",
            "sin_const_prop",
            "exp_const_prop",
            "expm1_const_prop",
            "tanh_const_prop",
            "logistic_const_prop",
            "conj_const_prop",
            "ceil_const_prop",
            "cbrt_const_prop",
            "real_const_prop",
            "imag_const_prop",
            "round_const_prop",
            "round_nearest_even_const_prop",
            "sign_const_prop",
            "floor_const_prop",
            "tan_const_prop",
            # binary constant propagation
            "add_const_prop",
            "and_const_prop",
            "atan2_const_prop",
            "complex_const_prop",
            "div_const_prop",
            "max_const_prop",
            "min_const_prop",
            "mul_const_prop",
            "or_const_prop",
            "pow_const_prop",
            "rem_const_prop",
            "sub_const_prop",
            "xor_const_prop",
            # other constant propagations
            "const_prop_through_barrier<16>",
            "concat_const_prop<1>($max_constant_threshold)",
            "dynamic_update_slice_const_prop($max_constant_threshold)",
            "clamp_const_prop",
        ],
    )

    if DUS_SLICE_SIMPLIFY[]
        push!(transform_passes_list, "dus_slice_simplify")
    end
    if SUM_TO_REDUCEWINDOW[]
        push!(transform_passes_list, "sum_to_reducewindow")
    end
    if SUM_TO_CONV[]
        push!(transform_passes_list, "sum_to_conv(0)")
    end
    if AGGRESSIVE_SUM_TO_CONV[]
        push!(transform_passes_list, "sum_to_conv(1)")
    end

    if WHILE_CONCAT[]
        push!(transform_passes_list, "while_concat")
        push!(transform_passes_list, "while_wrap")
        push!(transform_passes_list, "while_extend")
    end

    if dus_to_concat
        push!(transform_passes_list, "dus_to_concat")
    end

    if compile_options.reshape_propagate === :up
        append!(
            transform_passes_list,
            [
                "reshape_concat",
                "reshape_dus",
                "dot_reshape_pad<1>",
                "pad_dot_general<1>(0)",
                # XXX: see https://github.com/EnzymeAD/Enzyme-JAX/issues/1445
                # "pad_dot_general<1>(1)",
                "reshape_pad",
                "reshape_wrap",
                "reshape_rotate",
                "reshape_extend",
                "delete_dims_broadcast",
            ],
        )
        if AGGRESSIVE_PROPAGATION[]
            push!(transform_passes_list, "reshape_slice(0)")
            push!(transform_passes_list, "reshape_elementwise(0)")
            push!(transform_passes_list, "reshape_dynamic_slice(0)")
        else
            push!(transform_passes_list, "reshape_slice(1)")
            push!(transform_passes_list, "reshape_elementwise(1)")
            push!(transform_passes_list, "reshape_dynamic_slice(1)")
        end
    elseif compile_options.reshape_propagate === :down
        append!(
            transform_passes_list,
            [
                "concat_appending_reshape",
                "slice_reshape",
                "slice_reshape_slice<1>",
                "dynamic_slice_reshape_slice<1>",
                "slice_reshape_dynamic_slice<1>",
                "dynamic_slice_reshape_dynamic_slice<1>",
                "slice_reshape_concat<1>",
                "slice_reshape_elementwise<1>",
                "slice_reshape_dot_general<1>",
                "slice_reshape_pad<1>",
                "elementwise_reshape_like",
            ],
        )
        if AGGRESSIVE_PROPAGATION[]
            push!(transform_passes_list, "reshape_elementwise_only_fusible(0)")
        else
            push!(transform_passes_list, "reshape_elementwise_only_fusible(1)")
        end
    end

    if compile_options.transpose_propagate === :up
        append!(
            transform_passes_list,
            [
                "transpose_select",
                "transpose_while",
                "transpose_slice",
                "transpose_concat",
                "transpose_iota",
                "transpose_reduce",
                "transpose_reduce_window",
                "transpose_dus",
                "transpose_pad<1>",
                "transpose_einsum<1>",
                "transpose_wrap",
                "transpose_extend",
                "transpose_rotate",
                "transpose_dynamic_slice",
                "transpose_reverse",
                "transpose_batch_norm_training",
                "transpose_batch_norm_inference",
                "transpose_batch_norm_grad",
                "transpose_if",
                "transpose_fft",
                "transpose_reshape",
            ],
        )
        if AGGRESSIVE_PROPAGATION[]
            push!(transform_passes_list, "transpose_elementwise(0)")
        else
            push!(transform_passes_list, "transpose_elementwise(1)")
        end
    elseif compile_options.transpose_propagate === :down
        append!(
            transform_passes_list,
            [
                "reorder_elementwise_and_shape_op<16>",
                "elementwise_all_transpose_operands_simplify",
                "slice_transpose",
                "dynamic_slice_transpose",
                "einsum_transpose<1>",
                "slice_reshape_transpose<1>",
                "reduce_transpose_simplify",
                "reverse_transpose",
                "transpose_all_users_slice",
            ],
        )
    end

    if compile_options.no_nan
        append!(
            transform_passes_list,
            [
                "no_nan_compare_simplify(1)",
                "no_nan_self_sub_simplify(1)",
                "no_nan_add_sub_simplify(1)",
                "no_nan_mul_simplify(1)",
                "no_nan_div_simplify(1)",
            ],
        )
    else
        append!(
            transform_passes_list,
            [
                "no_nan_compare_simplify(0)",
                "no_nan_self_sub_simplify(0)",
                "no_nan_add_sub_simplify(0)",
                "no_nan_mul_simplify(0)",
                "no_nan_div_simplify(0)",
            ],
        )
    end

    if compile_options.all_finite
        append!(
            transform_passes_list,
            [
                "all_finite_is_finite",
                "all_finite_is_inf",
                "all_finite_is_pos_inf",
                "all_finite_is_neg_inf",
            ],
        )
    end

    lower_transform_passes = copy(transform_passes_list)

    if recognize_comms
        append!(
            transform_passes_list,
            [
                "recognize_extend",
                "recognize_wrap",
                "recognize_rotate",
                "recognize_updatewithoutcorners",
                "dusdus_to_dusextend",
            ],
        )
    end

    if lower_comms
        append!(
            lower_transform_passes,
            ["lower_extend", "lower_wrap", "lower_rotate", "lower_updatewithoutcorners"],
        )
    end

    transform_passes = join(
        [
            "enzyme-hlo-generate-td{" * join(transform_passes_list, ';') * "}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ],
        ",",
    )
    func_passes = join(["canonicalize", "cse", "canonicalize", transform_passes], ",")
    if lower_comms
        func_passes =
            func_passes *
            ",enzyme-hlo-generate-td{" *
            join(lower_transform_passes, ';') *
            "},transform-interpreter,enzyme-hlo-remove-transform"
    end
    if CONCATS_TO_DUS[]
        push!(
            transform_passes_list,
            "enzyme-hlo-generate-td{patterns=concat_to_onedim_dus},transform-interpreter,enzyme-hlo-remove-transform",
        )
    end
    passes = String[]
    if compile_options.inline
        push!(passes, "inline{default-pipeline=canonicalize max-iterations=4}")
    end
    if sroa
        push!(passes, "propagate-constant-bounds")
        if DUMP_LLVMIR[]
            push!(
                passes,
                "sroa-wrappers{dump_prellvm=true dump_postllvm=true instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
            )
        else
            push!(
                passes,
                "sroa-wrappers{instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
            )
        end
        push!(passes, "canonicalize")
        push!(
            passes,
            "sroa-wrappers{instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
        )
        push!(passes, "libdevice-funcs-raise")
        push!(passes, "canonicalize")
        push!(passes, "remove-duplicate-func-def")
    end
    push!(passes, func_passes)
    return join(passes, ',')
end

# TODO we want to be able to run the more advanced passes via transform dialect as an enzyme intermediate
# However, this errs as we cannot attach the transform with to the funcop itself [as we run a functionpass].
const enzyme_pass::String = "enzyme{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true}\"}"

function run_pass_pipeline!(mod, pass_pipeline, key=""; enable_verifier=true)
    pm = MLIR.IR.PassManager()
    MLIR.IR.enable_verifier!(pm, enable_verifier)
    opm = MLIR.IR.OpPassManager(pm)
    MLIR.IR.add_pipeline!(opm, pass_pipeline)
    MLIR.IR.run!(pm, mod, key)
    return mod
end

function run_pass_pipeline!(
    mod, propagation_options::ShardyPropagationOptions; enable_verifier=true
)
    pm = MLIR.IR.PassManager()
    MLIR.IR.enable_verifier!(pm, enable_verifier)
    opm = MLIR.IR.OpPassManager(pm)
    @ccall MLIR.API.mlir_c.addSdyPropagationPipeline(
        opm::MLIR.API.MlirOpPassManager,
        propagation_options.keep_sharding_rules::UInt8,
        propagation_options.conservative_propagation::UInt8,
        propagation_options.debug_sharding_origins::UInt8,
        propagation_options.debug_propagation_edge_sharding::UInt8,
        propagation_options.skip_convert_to_reshard::UInt8,
        propagation_options.skip_inline::UInt8,
        propagation_options.enable_insert_explicit_collectives::UInt8,
    )::Cvoid
    MLIR.IR.run!(pm, mod, "sdy_prop")
    return mod
end

const context_gc_vector = Dict{MLIR.IR.Context,Vector{Union{TracedRArray,TracedRNumber}}}()

# helper for debug purposes: String -> Text
function run_pass_pipeline_on_source(source, pass_pipeline; enable_verifier=true)
    return MLIR.IR.with_context() do ctx
        mod = parse(MLIR.IR.Module, source)
        run_pass_pipeline!(mod, pass_pipeline; enable_verifier)
        MLIR.IR.verifyall(MLIR.IR.Operation(mod); debug=true)
        Text(repr(mod))
    end
end

function __get_compile_options_and_kwargs(;
    compile_options::Union{Missing,CompileOptions}=missing,
    optimize::Union{Bool,Symbol,String}=true,
    no_nan::Bool=false,
    all_finite::Bool=false,
    inline::Bool=true,
    transpose_propagate::Symbol=:up,
    reshape_propagate::Symbol=:up,
    max_constant_threshold::Int=1024,
    raise::Union{Bool,String}=false,
    raise_first::Bool=false,
    legalize_chlo_to_stablehlo::Bool=false,
    cudnn_hlo_optimize::Bool=false,
    shardy_passes::Union{Symbol,ShardyPropagationOptions}=:post_sdy_propagation,
    optimize_then_pad::Bool=true,
    optimize_communications::Union{Bool,OptimizeCommunicationOptions}=true,
    assert_nonallocating::Bool=false,
    donated_args::Symbol=:auto,
    sync::Bool=false,
    kwargs...,
)
    return (
        Reactant.__compile_options_from_kwags(;
            compile_options,
            optimize,
            no_nan,
            all_finite,
            inline,
            transpose_propagate,
            reshape_propagate,
            max_constant_threshold,
            raise,
            raise_first,
            legalize_chlo_to_stablehlo,
            cudnn_hlo_optimize,
            shardy_passes,
            optimize_then_pad,
            optimize_communications,
            assert_nonallocating,
            donated_args,
            sync,
        ),
        kwargs,
    )
end

function compile_mlir(f, args; client=nothing, drop_unsupported_attributes=false, kwargs...)
    client = client !== nothing ? client : XLA.default_backend()
    backend = XLA.platform_name(client)

    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    results = MLIR.IR.with_context() do ctx
        mod = MLIR.IR.Module(MLIR.IR.Location())

        compile_options, kwargs = __get_compile_options_and_kwargs(; kwargs...)
        mlir_fn_res = compile_mlir!(
            mod,
            f,
            args,
            compile_options;
            backend,
            runtime=XLA.runtime(client),
            client,
            kwargs...,
        )

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        if drop_unsupported_attributes
            # Drop some of our attributes
            run_pass_pipeline!(mod, "drop-unsupported-attributes", "drop_enzymexla_attributes")
        end

        return mod, mlir_fn_res
    end

    return results
end

const PartitionKA = Ref{Bool}(true)

const cuindexBitWidth = Ref{Int}(32)
const cubinFormat = Ref{String}("bin")
const cuOptLevel = Ref{Int}(2)

# Wgatever the relevant highest version from our LLVM is within NVPTX.td
# Or more specifically looking at clang/lib/Driver/ToolChains/Cuda.cpp:684
#  We see relevant ptx version is CUDA 12.6 -> 85
#                                      12.2 -> 82
#                                      11.8 -> 78
function cubinFeatures()
    ver = @ccall MLIR.API.mlir_c.ReactantCudaDriverGetVersion()::UInt32
    # No cuda available
    if ver == 0
        return "+ptx86"
    end
    ver2 = @ccall MLIR.API.mlir_c.ReactantHermeticCudaGetVersion()::UInt32
    ver = min(ver, ver2)
    major, ver = divrem(ver, 1000)
    minor, patch = divrem(ver, 10)
    version = VersionNumber(major, minor, patch)
    # From https://github.com/llvm/llvm-project/blob/b60aed6fbabc291a7afbcb460453f9dcdce76f34/clang/lib/Driver/ToolChains/Cuda.cpp#L686
    cuver_map = Dict([
        (128, 87),
        (126, 85),
        (125, 85),
        (124, 84),
        (123, 83),
        (122, 82),
        (121, 81),
        (120, 80),
        (118, 78),
        (117, 77),
        (116, 76),
        (115, 75),
        (114, 74),
        (113, 73),
        (112, 72),
        (111, 71),
        (110, 70),
        (102, 65),
        (101, 64),
        (100, 63),
        (92, 61),
        (91, 61),
        (90, 60),
    ])
    mver = major * 10 + minor
    if !in(mver, keys(cuver_map))
        return 86
    end
    ptx = cuver_map[mver]
    return "+ptx$ptx"
end

const DEBUG_KERNEL = Ref{Bool}(false)
const DUMP_LLVMIR = Ref{Bool}(false)
const DUMP_FAILED_LOCKSTEP = Ref{Bool}(false)
const OpenMP = Ref{Bool}(true)
const SROA_ATTRIBUTOR = Ref{Bool}(true)

function activate_raising!(is_raising::Bool)
    stack = get!(task_local_storage(), :reactant_is_raising) do
        Bool[]
    end
    push!(stack, is_raising)
    return nothing
end

function deactivate_raising!(is_raising::Bool)
    key = :reactant_is_raising
    is_raising === last(task_local_storage(key)) ||
        error("Deactivating wrong Reactant raising context")
    return pop!(task_local_storage(key))
end

function raising(; throw_error::Bool=true)
    key = :reactant_is_raising
    if !(haskey(task_local_storage(), key) && !Base.isempty(task_local_storage(key)))
        throw_error && error("No Reactant raising context")
    end
    return last(task_local_storage(key)::Vector{Bool})
end

function raising!(f, is_raising::Bool)
    activate_raising!(is_raising)
    try
        return f()
    finally
        deactivate_raising!(is_raising)
    end
end

function get_optimize_comms_passes(options::Bool)
    if !options
        return [
            "enzyme-hlo-generate-td{patterns=lower_rotate;lower_wrap;lower_extend;lower_updatewithoutcorners}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ]
    end
    return get_optimize_comms_passes(OptimizeCommunicationOptions())
end

function get_optimize_comms_passes(options::OptimizeCommunicationOptions)
    options_str = String(options)
    res = [
        "enzyme-hlo-generate-td{patterns=concat_to_onedim_dus;concat_to_onedim_dusslice;concatreshape_to_onedim_dus}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        "enzyme-hlo-generate-td{patterns=reshape_to_broadcast}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        options_str,
        "enzyme-hlo-generate-td{patterns=lower_rotate;lower_wrap;lower_extend;lower_updatewithoutcorners}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        options_str,
    ]
    return res
end

function get_stablehlo_to_hlo_passes(; stablehlo_to_mhlo::Bool=true)
    passes = (
        "func.func(stablehlo-ext-chlo-recompose-ops)",
        "symbol-dce",
        "func.func(chlo-legalize-to-high-level-mhlo)",
        "func.func(chlo-legalize-to-stablehlo)",
    )
    if stablehlo_to_mhlo
        passes = (passes..., "stablehlo-legalize-to-hlo")
    end
    passes = (
        passes..., "canonicalize", "func.func(stablehlo-ext-sink-constants-to-control-flow)"
    )
    return passes
end

function compile_mlir!(
    mod,
    f,
    args,
    compile_options::CompileOptions,
    callcache=default_callcache(),
    sdycache=default_sdycache(),
    sdygroupidcache=default_sdygroupidcache();
    fn_kwargs=(),
    backend="gpu",
    runtime::Union{Val{:PJRT},Val{:IFRT}},
    legalize_stablehlo_to_mhlo::Bool=false,
    client=nothing,
    kwargs...,
)
    client = client !== nothing ? client : XLA.default_backend()

    # Explicitly don't use block! to avoid creating a closure, which creates
    # both compile-time and relocatability issues

    MLIR.IR.activate!(mod)
    MLIR.IR.activate!(MLIR.IR.body(mod))
    activate_callcache!(callcache)
    activate_sdycache!(sdycache)
    activate_sdygroupidcache!(sdygroupidcache)

    # Save in the TLS whether we are raising.  We identify that condition by
    # checking whether the user set an explicit list of passes, or chose
    # `raise=true` to use the default passes.
    raise = compile_options.raise
    if backend == "tpu" && raise isa Bool
        raise = true
    end
    is_raising = raise isa String || raise
    activate_raising!(is_raising)

    fnname = string(f)
    mlir_fn_res = try
        Reactant.TracedUtils.make_mlir_fn(
            f,
            args,
            fn_kwargs,
            fnname,
            true;
            runtime,
            compile_options.optimize_then_pad,
            kwargs...,
        )
    finally
        deactivate_raising!(is_raising)
        deactivate_sdycache!(sdycache)
        deactivate_sdygroupidcache!(sdygroupidcache)
        deactivate_callcache!(callcache)
        MLIR.IR.deactivate!(MLIR.IR.body(mod))
        MLIR.IR.deactivate!(mod)
    end
    (;
        fnwrapped,
        traced_result,
        seen_args,
        ret,
        linear_args,
        skipped_args,
        in_tys,
        linear_results,
        skipped_results,
        is_sharded,
    ) = mlir_fn_res
    compiled_f = mlir_fn_res.f

    # Custom Kernels without Raising will lead to suboptimal codegen for is_sharded, force
    # raising
    if is_sharded
        is_raising = true
        raise isa Bool && (raise = true)
    end

    toolkit = XLA.CUDA_DATA_DIR[]

    if backend == "cpu" || backend == "tpu"
        kern = "lower-kernel{backend=cpu},canonicalize"
        if backend == "tpu"
            jit = "lower-jit{openmp=$(OpenMP[]) backend=cpu},symbol-dce,strip-debuginfo"
        else
            jit = "lower-jit{openmp=$(OpenMP[]) backend=cpu},symbol-dce"
        end
    else
        kern = if is_raising
            "lower-kernel{backend=cpu},symbol-dce,canonicalize"
        else
            "lower-kernel,canonicalize"
        end

        device_properties = XLA.device_properties(XLA.default_device(client))
        cubinChip = "sm_$(device_properties.major)$(device_properties.minor)"

        if DEBUG_KERNEL[]
            curesulthandler = dlsym(
                Reactant_jll.libReactantExtra_handle, "ReactantHandleCuResult"
            )
            @assert curesulthandler !== nothing
            curesulthandler = Base.reinterpret(UInt, curesulthandler)
            extra_lowerjit_options = "debug=true cuResultHandlerPtr=$curesulthandler "
        else
            extra_lowerjit_options = ""
        end
        jit = "lower-jit{$(extra_lowerjit_options)cuOptLevel=$(cuOptLevel[]) cubinFormat=$(cubinFormat[]) indexBitWidth=$(cuindexBitWidth[])  cubinChip=$(cubinChip) cubinFeatures=$(cubinFeatures()) run_init=true toolkitPath=$toolkit},symbol-dce"
    end

    recognize_comms = true
    lower_comms = true
    if is_sharded && (
        compile_options.shardy_passes == :to_mhlo_shardings ||
        compile_options.shardy_passes == :post_sdy_propagation ||
        compile_options.shardy_passes isa ShardyPropagationOptions
    )
        lower_comms = false
    end

    opt_passes = optimization_passes(
        compile_options; sroa=true, recognize_comms, lower_comms, backend, is_sharded
    )
    opt_passes2 = optimization_passes(
        compile_options; sroa=false, recognize_comms, lower_comms, backend, is_sharded
    )

    raise_passes = if raise isa String
        # Raising passes were specified
        raise
    elseif raise

        # Raise enabled but use default passes
        # TODO remove redundant libdevice raise after fixing phase ordering
        result =
            "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=$(DUMP_FAILED_LOCKSTEP[])},canonicalize,arith-raise{stablehlo=true}," *
            opt_passes2

        if DUS_TO_CONCAT[]
            opt_passes3 = optimization_passes(
                compile_options;
                sroa=false,
                dus_to_concat=true,
                recognize_comms,
                lower_comms,
                backend,
                is_sharded,
            )
            result = result * "," * opt_passes3
        end
        result
    else
        "canonicalize"
    end

    blas_int_width = sizeof(BlasInt) * 8
    lower_enzymexla_linalg_pass = "lower-enzymexla-linalg{backend=$backend \
                                   blas_int_width=$blas_int_width},\
                                   lower-enzymexla-blas{backend=$backend \
                                   blas_int_width=$blas_int_width},\
                                   lower-enzymexla-lapack{backend=$backend \
                                   blas_int_width=$blas_int_width}"

    legalize_chlo_to_stablehlo =
        if legalize_stablehlo_to_mhlo || compile_options.legalize_chlo_to_stablehlo
            get_stablehlo_to_hlo_passes(; stablehlo_to_mhlo=legalize_stablehlo_to_mhlo)
        else
            ()
        end

    legal_to_run_shardy_passes = compile_options.optimization_passes === :all

    if compile_options.optimization_passes === :all
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                end,
                ",",
            ),
            "all",
        )
    elseif compile_options.optimization_passes === :before_kernel
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    ["mark-func-memory-effects", opt_passes]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                    ]
                end,
                ',',
            ),
            "before_kernel",
        )
    elseif compile_options.optimization_passes === :before_jit
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                    ]
                end,
                ',',
            ),
            "before_jit",
        )
    elseif compile_options.optimization_passes === :before_raise
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    ["mark-func-memory-effects", opt_passes]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                    ]
                end,
                ',',
            ),
            "before_raise",
        )
    elseif compile_options.optimization_passes === :no_enzyme
        run_pass_pipeline!(
            mod,
            join(
                [
                    "mark-func-memory-effects",
                    opt_passes,
                    "enzyme-batch",
                    opt_passes2,
                    enzyme_pass,
                    opt_passes2,
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    legalize_chlo_to_stablehlo...,
                    opt_passes2,
                ],
                ',',
            ),
            "no_enzyme",
        )
    elseif compile_options.optimization_passes === :only_enzyme
        run_pass_pipeline!(
            mod,
            join(
                [
                    "mark-func-memory-effects",
                    "enzyme-batch",
                    enzyme_pass,
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                ],
                ',',
            ),
            "only_enzyme",
        )
    elseif compile_options.optimization_passes === :after_enzyme
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        enzyme_pass,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        "enzyme-batch",
                        enzyme_pass,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                end,
                ',',
            ),
            "after_enzyme",
        )
    elseif compile_options.optimization_passes === :before_enzyme
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math",
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math",
                        kern,
                        raise_passes,
                        lower_enzymexla_linalg_pass,
                        jit,
                    ]
                end,
                ',',
            ),
            "before_enzyme",
        )
    elseif compile_options.optimization_passes === :canonicalize
        run_pass_pipeline!(mod, "mark-func-memory-effects,canonicalize", "canonicalize")
    elseif compile_options.optimization_passes === :just_batch
        run_pass_pipeline!(mod, "enzyme-batch", "enzyme-batch")
    elseif compile_options.optimization_passes isa String
        run_pass_pipeline!(mod, compile_options.optimization_passes, "custom_pass")
    end

    if compile_options.optimization_passes isa Symbol &&
        compile_options.optimization_passes === :all &&
        (
            compile_options.transpose_propagate === :up ||
            compile_options.reshape_propagate === :up
        )
        # We tried propagating reshapes and transposes up. If at this point we are left
        # with them, we propagate them down to minimize the number of Ops in the IR.
        # Since this might enable certain raising, we do push down -> push up -> push down
        common_kwargs = (;
            recognize_comms,
            lower_comms,
            backend,
            is_sharded,
            raise_shlo_to_blas_lapack=false,
        )
        opt_passes_down = optimization_passes(
            Reactant.__compile_options_with_reversed_propagation(compile_options);
            common_kwargs...,
        )
        opt_passes_up = optimization_passes(compile_options; common_kwargs...)
        run_pass_pipeline!(
            mod,
            join([opt_passes_down, opt_passes_up, opt_passes_down], ","),
            "post_op_transpose_reshape",
        )
    end

    if backend == "cuda" && compile_options.cudnn_hlo_optimize
        run_pass_pipeline!(mod, "enzymexla-cudnn-hlo-opt", "cudnn-hlo-opt")
    end

    # Now we resolve paddings if `optimize_then_pad`
    if compile_options.optimize_then_pad
        padded_inputs = IdDict()
        has_padded_inputs = false
        for (k, v) in seen_args
            v isa Reactant.TracedType || continue
            if Reactant.has_padding(k)
                has_padded_inputs = true
                padded_inputs[v] = Reactant.get_padding(k)
            end
        end

        if has_padded_inputs
            MLIR.IR.DUMP_MLIR_ALWAYS[] && MLIR.IR.dump_mlir(mod, nothing, "pre_padding")

            in_tys_padded = Vector{MLIR.IR.Type}(undef, length(linear_args))
            input_arg_padded_idxs = Int[]
            for (i, arg) in enumerate(linear_args)
                if haskey(padded_inputs, arg)
                    push!(input_arg_padded_idxs, i)
                    in_tys_padded[i] = MLIR.IR.TensorType(
                        collect(Int, reverse(size(arg) .+ padded_inputs[arg])),
                        MLIR.IR.Type(Reactant.unwrapped_eltype(arg)),
                    )
                else
                    in_tys_padded[i] = in_tys[i]
                end
            end

            out_tys_padded = Vector{MLIR.IR.Type}(undef, length(linear_results))
            output_res_padded_idxs = Int[]
            for (i, res) in enumerate(linear_results)
                if haskey(padded_inputs, res)
                    push!(output_res_padded_idxs, i)
                    out_tys_padded[i] = MLIR.IR.TensorType(
                        collect(Int, reverse(size(res) .+ padded_inputs[res])),
                        MLIR.IR.Type(Reactant.unwrapped_eltype(res)),
                    )
                else
                    out_tys_padded[i] = Reactant.TracedUtils.transpose_ty(
                        Reactant.Ops.mlir_type(res)
                    )
                end
            end

            fnname_old = fnname
            fnname = string(f, "_padded")
            func_with_padding = MLIR.Dialects.func.func_(;
                sym_name=fnname,
                function_type=MLIR.IR.FunctionType(in_tys_padded, out_tys_padded),
                arg_attrs=MLIR.IR.attr(compiled_f, "arg_attrs"),
                res_attrs=MLIR.IR.attr(compiled_f, "res_attrs"),
                no_inline=MLIR.IR.attr(compiled_f, "no_inline"),
                body=MLIR.IR.Region(),
                sym_visibility=MLIR.IR.attr(compiled_f, "private"),
            )
            fnbody = MLIR.IR.Block(
                in_tys_padded,
                [
                    MLIR.IR.Location(
                        MLIR.API.mlirValueGetLocation(
                            MLIR.IR.argument(
                                MLIR.IR.first_block(MLIR.IR.region(compiled_f, 1)), i
                            ),
                        ),
                    ) for i in 1:length(linear_args)
                ],
            )
            push!(MLIR.IR.region(func_with_padding, 1), fnbody)
            MLIR.IR.activate!(fnbody)
            push!(MLIR.IR.body(mod), func_with_padding)

            try
                call_args = MLIR.IR.Value[
                    MLIR.IR.argument(fnbody, i) for i in 1:length(linear_args)
                ]

                for i in input_arg_padded_idxs
                    arg = linear_args[i]
                    padding = padded_inputs[arg]

                    block_arg = MLIR.IR.argument(fnbody, i)
                    unpad_op = Reactant.TracedUtils.unpad_val_op(
                        block_arg, reverse(padding), reverse(size(arg) .+ padding)
                    )

                    call_args[i] = MLIR.IR.result(unpad_op, 1)
                end

                ftype = MLIR.IR.Type(MLIR.IR.attr(compiled_f, "function_type"))
                call_op = MLIR.Dialects.func.call(
                    call_args;
                    result_0=[MLIR.IR.result(ftype, i) for i in 1:MLIR.IR.nresults(ftype)],
                    callee=MLIR.IR.FlatSymbolRefAttribute(fnname_old),
                )

                results = MLIR.IR.Value[
                    MLIR.IR.result(call_op, i) for i in 1:MLIR.IR.nresults(call_op)
                ]

                for i in output_res_padded_idxs
                    res = linear_results[i]
                    padding = padded_inputs[res]

                    pad_op = MLIR.Dialects.stablehlo.pad(
                        results[i],
                        Reactant.TracedUtils.promote_to(
                            TracedRNumber{Reactant.unwrapped_eltype(res)}, 0
                        ).mlir_data;
                        edge_padding_low=MLIR.IR.DenseArrayAttribute(
                            fill(0, length(padding))
                        ),
                        edge_padding_high=MLIR.IR.DenseArrayAttribute(
                            collect(reverse(padding))
                        ),
                        interior_padding=MLIR.IR.DenseArrayAttribute(
                            fill(0, length(padding))
                        ),
                    )

                    results[i] = MLIR.IR.result(pad_op, 1)
                end

                MLIR.Dialects.func.return_(results)
            finally
                MLIR.IR.deactivate!(fnbody)
            end

            # we just need the ops to potentially remove slices / paddings
            if compile_options.optimization_passes === :all
                run_pass_pipeline!(
                    mod,
                    join(
                        [
                            opt_passes,
                            "canonicalize",
                            "cse",
                            "canonicalize",
                            opt_passes2,
                            lower_enzymexla_linalg_pass,
                            jit,
                        ],
                        ",",
                    ),
                    "mid_pad_opts",
                )
            end

            MLIR.IR.attr!(compiled_f, "sym_visibility", MLIR.IR.Attribute("private"))
            run_pass_pipeline!(
                mod,
                "inline{default-pipeline=canonicalize max-iterations=4}",
                "inline_pad_opts",
            )

            compiled_f = func_with_padding
            in_tys = in_tys_padded
        end
    end

    # shardy passes
    use_shardy_partitioner = false
    result_shardings = missing
    if is_sharded && legal_to_run_shardy_passes
        module_op = copy(MLIR.IR.Operation(mod))
        mod_copied = MLIR.IR.Module(module_op)

        if compile_options.shardy_passes isa ShardyPropagationOptions
            run_pass_pipeline!(mod_copied, compile_options.shardy_passes)
            run_pass_pipeline!(mod_copied, "sdy-close-shardings", "sdy_close_shardings")
        else
            run_pass_pipeline!(
                mod_copied,
                join(["sdy-propagation-pipeline", "sdy-close-shardings"], ","),
                "sdy_prop_capture_res_shardings",
            )
        end

        func_op = MLIR.API.mlirSymbolTableLookup(MLIR.IR.SymbolTable(module_op), fnname)
        @assert func_op.ptr !== C_NULL
        func_op_new_module = MLIR.IR.Operation(func_op, false)

        result_attrs = MLIR.IR.attr(func_op_new_module, "res_attrs")
        if result_attrs !== nothing
            result_shardings = Vector{Union{Sharding.NamedSharding,Sharding.Replicated}}(
                undef, length(result_attrs)
            )
            for i in 1:length(result_attrs)
                result_shardings[i] = Sharding.sdy_sharding_to_reactant_sharding(
                    result_attrs[i - 1], mlir_fn_res.global_device_ids, mod_copied
                )
            end
        else
            result_shardings = [Sharding.Replicated() for _ in 1:length(linear_results)]
        end

        if compile_options.shardy_passes === :none
            use_shardy_partitioner = true
        elseif compile_options.shardy_passes === :post_sdy_propagation
            use_shardy_partitioner = true
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-propagation-pipeline",
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "func.func(sdy-reshard-to-collectives)",
                    ],
                    ",",
                ),
                "post_sdy_propagation",
            )
        elseif compile_options.shardy_passes isa ShardyPropagationOptions
            run_pass_pipeline!(mod, compile_options.shardy_passes)
            # sdy passes are run deep inside the XLA compiler. So the only way to respect
            # the options is to export them to MHLO shardings
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "xla-sdy-stablehlo-export-pipeline",
                    ],
                    ",",
                ),
                "sdy_export",
            )
        elseif compile_options.shardy_passes === :to_mhlo_shardings
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-propagation-pipeline",
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "func.func(sdy-reshard-to-collectives)",
                        "xla-sdy-stablehlo-export-pipeline",
                    ],
                    ",",
                ),
                "to_mhlo_shardings",
            )
        end
    end

    run_pass_pipeline!(mod, "mark-func-memory-effects", "mark-func-memory-effects")

    func_op = MLIR.API.mlirSymbolTableLookup(
        MLIR.IR.SymbolTable(MLIR.IR.Operation(mod)), fnname
    )
    @assert func_op.ptr !== C_NULL
    func_op = MLIR.IR.Operation(func_op, false)
    fnbody = MLIR.IR.first_block(MLIR.IR.region(func_op, 1))::MLIR.IR.Block
    ret = MLIR.IR.terminator(fnbody)::MLIR.IR.Operation

    preserved_args = Tuple{TracedType,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedType[]
    results_mask = falses(length(results))

    for (i, op) in enumerate(results)
        if !MLIR.IR.is_block_arg(op) ||
            !Reactant.TracedUtils.has_idx(linear_results[i], :args) # new buffer
            push!(nresults, op)
            push!(linear_results2, linear_results[i])
            results_mask[i] = true
            continue
        end
        push!(preserved_args, (linear_results[i], MLIR.IR.block_arg_num(op)))
    end

    MLIR.API.mlirOperationDestroy(ret.operation)
    ret.operation = MLIR.API.MlirOperation(C_NULL)
    MLIR.IR.block!(fnbody) do
        return MLIR.Dialects.func.return_(nresults)
    end

    out_tys2 = [MLIR.IR.type(a) for a in nresults]

    res_attrs = MLIR.IR.attr(compiled_f, "res_attrs")
    if res_attrs isa MLIR.IR.Attribute
        res_attrs = MLIR.IR.Attribute[
            res_attrs[i - 1] for (i, present) in enumerate(results_mask) if present
        ]
    end

    if result_shardings !== missing
        result_shardings_after_masking = eltype(result_shardings)[]
        for (i, present) in enumerate(results_mask)
            if present
                push!(result_shardings_after_masking, result_shardings[i])
            end
        end
    else
        result_shardings_after_masking = missing
    end

    func3 = MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=MLIR.IR.FunctionType(in_tys, out_tys2),
        arg_attrs=MLIR.IR.attr(compiled_f, "arg_attrs"),
        res_attrs,
        no_inline=MLIR.IR.attr(compiled_f, "no_inline"),
        body=MLIR.IR.Region(),
    )
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func3, 1), MLIR.IR.region(compiled_f, 1))

    push!(MLIR.IR.body(mod), func3)

    mem = MLIR.IR.attr(compiled_f, "enzymexla.memory_effects")
    if !(mem isa Nothing)
        MLIR.IR.attr!(func3, "enzymexla.memory_effects", mem)
    end

    MLIR.API.mlirOperationDestroy(compiled_f.operation)
    compiled_f.operation = MLIR.API.MlirOperation(C_NULL)

    # Add a `donated` attr to the function arguments. This doesn't affect XLA, but lets us
    # check which arguments were donated.
    preserved_args_idx = last.(preserved_args)
    donated_args_mask = Vector{Bool}(undef, length(linear_args))
    for (i, arg) in enumerate(linear_args)
        if compile_options.donated_args == :auto
            if (i - 1) ∉ preserved_args_idx
                donated_args_mask[i] = true

                residx = findfirst(Base.Fix1(===, arg), linear_results2)
                if residx !== nothing
                    MLIR.API.mlirFuncSetArgAttr(
                        func3,
                        i - 1,
                        "tf.aliasing_output",
                        MLIR.IR.Attribute(Int32(residx - 1)),
                    )
                end
            else
                donated_args_mask[i] = false
            end
        else # :none
            donated_args_mask[i] = false
        end
    end

    # drop certain operations from the module if using TPU backend
    if backend == "tpu"
        for op in collect(MLIR.IR.OperationIterator(MLIR.IR.body(mod)))
            if MLIR.IR.dialect(op) == :llvm
                MLIR.API.mlirOperationDestroy(op.operation)
                op.operation = MLIR.API.MlirOperation(C_NULL)
            end
        end
    end

    if compile_options.assert_nonallocating
        if length(linear_args) - length(preserved_args_idx) != length(nresults)
            str = sprint() do io
                Base.show(IOContext(io, :debug => true), func3)
            end
            throw(
                AssertionError(
                    """length(preserved_args_idx) = $(length(preserved_args_idx))
             donated = length(linear_args) - length(preserved_args_idx) = $(length(linear_args) - length(preserved_args_idx))
                    length(nresults) = $(length(nresults))
                    linear_args = $linear_args
                    linear_results = $linear_results
                    $((MLIR.IR.argument(fnbody, i) for i in 1:length(in_tys))...)
                    preserved_args = $(preserved_args_idx)
                    $str
                    """,
                ),
            )
        end
    end

    concrete_result = make_tracer(
        OrderedIdDict(), traced_result, ("result",), TracedToConcrete; runtime
    )

    return Reactant.TracedUtils.CompiledMlirFnResult(
        fnwrapped,
        func3,
        traced_result,
        mlir_fn_res.result,
        seen_args,
        ret,
        linear_args,
        skipped_args,
        in_tys,
        linear_results2,
        skipped_results,
        mlir_fn_res.num_partitions,
        mlir_fn_res.num_replicas,
        mlir_fn_res.is_sharded,
        preserved_args,
        concrete_result,
        mlir_fn_res.unique_meshes,
        mlir_fn_res.mutated_args,
        use_shardy_partitioner,
        result_shardings_after_masking,
        mlir_fn_res.global_device_ids,
        donated_args_mask,
        Reactant.TracedUtils.is_pure(func3),
    )
end

function get_common_compile_options()
    return Dict{Symbol,Any}(
        :optimize => true,
        :no_nan => false,
        :client => nothing,
        :raise => false,
        :raise_first => false,
        :shardy_passes => :(:post_sdy_propagation),
        :assert_nonallocating => false,
        :donated_args => :(:auto),
        :transpose_propagate => :(:up),
        :reshape_propagate => :(:up),
        :optimize_then_pad => true,
        :optimize_communications => true,
        :cudnn_hlo_optimize => false,
        :legalize_chlo_to_stablehlo => false,
        :compile_options => missing,
    )
end

const COMMON_COMPILE_OPTIONS_DOCS = """
  - `compile_options`: If provided, then all other compilation options will be ignored.
    This should be an object of type [`CompileOptions`](@ref).
  - `optimize`: This option maps to the `optimization_passes` field of
    [`CompileOptions`](@ref). See the documentation of `CompileOptions` for more details.
  - `client`: XLA Client used for compilation. If not specified, the default client is used.

For details about other compilation options see the documentation of
[`CompileOptions`](@ref).
"""

const SYNC_DOCS = """
  - `sync`: Reactant computations are asynchronous by default. If `true`, the computation
    will be executed synchronously, blocking till the computation is complete. This is
    recommended when benchmarking.
"""

"""
    @code_hlo [optimize = ...] [no_nan = <true/false>] f(args...)

Prints the compiled MLIR module for the function `f` with arguments `args`.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@code_xla`](@ref), [`@code_mhlo`](@ref).
"""
macro code_hlo(args...)
    compile_expr, (; compiled) = compile_call_expr(
        __module__,
        compile_mlir,
        merge(get_common_compile_options(), Dict{Symbol,Any}(:shardy_passes => :(:none))),
        args...,
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(first)($(compiled))
        )
    )
    #! format: on
end

"""
    @code_mhlo [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to `@code_hlo`, but runs additional passes to export the stablehlo module to MHLO.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@code_xla`](@ref), [`@code_hlo`](@ref).
"""
macro code_mhlo(args...)
    compile_expr, (; compiled) = compile_call_expr(
        __module__,
        compile_mlir,
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(
                :legalize_stablehlo_to_mhlo => true, :shardy_passes => :(:to_mhlo_shardings)
            ),
        ),
        args...,
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(first)($(compiled))
        )
    )
    #! format: on
end

"""
    @code_xla [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to [`@code_hlo`](@ref), but runs additional XLA passes and exports MLIR to XLA HLO.
This is the post optimizations XLA HLO module.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)
  - `before_xla_optimizations`: If `true`, return the `before_optimizations` HLO module.

See also [`@code_mhlo`](@ref), [`@code_hlo`](@ref).
"""
macro code_xla(args...)
    compile_expr, (; compiled) = compile_call_expr(
        __module__,
        compile_xla,
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(:before_xla_optimizations => false),
        ),
        args...,
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(compiled)[3]
        )
    )
    #! format: on
end

"""
    @compile [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Compile the function `f` with arguments `args` and return the compiled function.

## Note

Note that `@compile foo(bar(x))` is equivalent to
```julia
y = bar(x)  # first compute the output of `bar(x)`, say `y`
@compile foo(y) # then compile `foo` for `y`
```
That is, like `@jit`, `@compile` only applies to the outermost function call; it does *not* compile the composed function `foo(bar(x))` jointly.
Hence, if you want to compile the composed function `foo(bar(x))` jointly, you need to introduce an intermediate function, i.e.,
```julia
baz(x) = foo(bar(x))
@compile baz(x)
```

## Options

$(SYNC_DOCS)
$(COMMON_COMPILE_OPTIONS_DOCS)
  - `serializable`: If `true`, the compiled function will be serializable. This is needed
    for saving the compiled function to disk and loading it later. Defaults to `false`.

See also [`@jit`](@ref), [`@code_hlo`](@ref), [`@code_mhlo`](@ref), [`@code_xla`](@ref).
"""
macro compile(args...)
    default_options = merge(
        get_common_compile_options(),
        Dict{Symbol,Any}(:sync => false, :serializable => false),
    )
    return esc(first(compile_call_expr(__module__, compile, default_options, args...)))
end

"""
    @jit [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Run @compile f(args..) then immediately execute it. Most users should use [`@compile`](@ref)
instead to cache the compiled function and execute it later.

## Note

Note that `@jit foo(bar(x))` is equivalent to
```julia
y = bar(x)  # first compute the output of `bar(x)`, say `y`
@jit foo(y) # then compile `foo` for `y` and execute it
```
That is, like `@compile`, `@jit` only applies to the outermost function call; it does *not* compile the composed function `foo(bar(x))` jointly.
Hence, if you want to compile the composed function `foo(bar(x))` jointly, you need to introduce an intermediate function, i.e.,
```julia
baz(x) = foo(bar(x))
@jit baz(x)
```

## Options

$(SYNC_DOCS)
$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@compile`](@ref), [`@code_hlo`](@ref), [`@code_mhlo`](@ref), [`@code_xla`](@ref).
"""
macro jit(args...)
    default_options = merge(get_common_compile_options(), Dict{Symbol,Any}(:sync => false))
    compile_expr, (; compiled, args) = compile_call_expr(
        __module__, compile, default_options, args...
    )
    #! format: off
    return esc(
        :(
            $(compile_expr);
            $(compiled)($(args)...)
        )
    )
    #! format: on
end

function compile_call_expr(mod, compiler, options::Dict, args...)
    while length(args) > 1
        option, args = args[1], args[2:end]
        if !Meta.isexpr(option, :(=))
            error("Invalid option $(option)")
        else
            option_name = option.args[1]
            @assert haskey(options, option_name) "Invalid option name '$(option_name)'. Valid options are $(join(keys(options), ", "))"
            options[option_name] = option.args[2]
        end
    end

    call = only(args)
    f_symbol = gensym(:f)
    args_symbol = gensym(:args)
    kwargs_symbol = gensym(:kwargs)
    compiled_symbol = gensym(:compiled)

    if Meta.isexpr(call, :call)
        bcast, fname, fname_full = correct_maybe_bcast_call(call.args[1])
        fname = if bcast
            quote
                if isdefined(mod, $(Meta.quot(fname_full)))
                    $(fname_full)
                else
                    Base.Broadcast.BroadcastFunction($(fname))
                end
            end
        else
            :($(fname))
        end
        args_rhs = call.args[2:end]

        # if (;) is used, we need to extract the kwargs
        if length(args_rhs) ≥ 1 && Meta.isexpr(args_rhs[1], :parameters)
            kwargs_rhs = args_rhs[1].args
            args_rhs = args_rhs[2:end]
        else
            kwargs_rhs = ()
        end
        kw_idxs = findall(Base.Fix2(Meta.isexpr, :kw), args_rhs)
        arg_idxs = setdiff(1:length(args_rhs), kw_idxs)

        kwargs_rhs = (kwargs_rhs..., args_rhs[kw_idxs]...)
        args_rhs = Expr(:tuple, args_rhs[arg_idxs]...)
    elseif Meta.isexpr(call, :(.), 2) && Meta.isexpr(call.args[2], :tuple)
        fname = :($(Base.Broadcast.BroadcastFunction)($(call.args[1])))
        args_rhs = only(call.args[2:end])
        kwargs_rhs = ()
    else
        error("Invalid function call: $(call)")
    end

    return (
        quote
            $(f_symbol) = $(fname)
            $(args_symbol) = $(args_rhs)
            $(kwargs_symbol) = (; $(kwargs_rhs...))
            $(compiled_symbol) = $(compiler)(
                $(f_symbol),
                $(args_symbol);
                fn_kwargs=$(kwargs_symbol),
                $(Expr.(:kw, keys(options), values(options))...),
            )
        end,
        (; compiled=compiled_symbol, args=args_symbol),
    )
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
    size_x,
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

The _linearized arguments_ do not directly refer to the  are the arguments that have been flattened into a single list.
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

    if is_sharded
        inv_seen_args = Reactant.OrderedIdDict()
        for (k, v) in seen_args
            inv_seen_args[v] = k
        end
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

# XXX: Currently we copy to host and then make the transfer to the sharded devices. This is
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
                    # XXX: we might need to handle sharding here
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
                error("TODO: Not yet Implemented. Open an issue on Reactant.jl.")
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
    for (result, arg_idx) in preserved_args
        paths = (
            (
                p for p in Reactant.TracedUtils.get_paths(result) if
                length(p) > 0 && (p[1] == :resargs || p[1] == :args)
            )...,
        )

        for path in paths
            arg = linear_args[arg_idx + 1]
            argpath = only((
                p for
                p in Reactant.TracedUtils.get_paths(arg) if length(p) > 0 && p[1] == :args
            ))

            @assert path[1] == :resargs || path[1] == :args "Expected :resargs or :args, got $(path[1])"
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
                error("TODO: Not yet Implemented. Open an issue on Reactant.jl.")
            end

            argres = :(args[$(argpath[2])])
            for p in argpath[3:end]
                argres = :(traced_getfield($argres, $(Meta.quot(p))))
            end

            res = :(traced_setfield!($res, :data, $argres.data, $(path)))
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

function __add_mhlo_attributes_and_name!(mod::MLIR.IR.Module, f; kwargs...)
    fname = string(f)
    length(fname) > 10 && (fname = fname[1:7] * "...")
    __add_mhlo_attributes_and_name!(mod, fname; kwargs...)
    return nothing
end

function __add_mhlo_attributes_and_name!(
    mod::MLIR.IR.Module, fname::String; num_partitions=1, num_replicas=1
)
    moduleop = MLIR.IR.Operation(mod)
    module_name = Reactant.TracedUtils.__lookup_unique_name_in_module(
        mod, "reactant_" * fname
    )
    module_name = MLIR.IR.Attribute(module_name)
    MLIR.IR.attr!(moduleop, "mhlo.num_partitions", MLIR.IR.Attribute(num_partitions))
    MLIR.IR.attr!(moduleop, "mhlo.num_replicas", MLIR.IR.Attribute(num_replicas))
    MLIR.IR.attr!(
        moduleop, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName()), module_name
    )
    return nothing
end

function __resolve_device_and_client(client, seen_args, linear_args, is_sharded)
    if is_sharded
        client === nothing && (client = XLA.default_backend())
        return client, nothing
    end

    device = nothing
    if length(linear_args) > 0
        devices_list = []
        for (k, v) in seen_args
            !(v isa TracedRArray || v isa TracedRNumber) && continue
            buffer = k.data isa Tuple ? only(k.data) : k.data
            push!(devices_list, XLA.device(buffer))
        end
        if !isempty(devices_list)
            if !allequal(devices_list)
                msg = "Expected all arguments to be on the same device, got:\n"
                for (i, device) in enumerate(unique(devices_list))
                    msg *= "    Device $(i): $(string(device))\n"
                end
                throw(ArgumentError(msg))
            end
            @assert allequal(devices_list) "All arguments must be on the same device: $(devices_list)"
            device = first(devices_list)
        end
    end

    if client === nothing
        if device !== nothing
            client = XLA.client(device)
        else
            client = XLA.default_backend()
            device = XLA.default_device(client)
        end
    else
        if device !== nothing
            @assert client == XLA.client(device) "client ($(client)) and XLA.client(device) ($(XLA.client(device))) must be the same"
        else
            device = XLA.default_device(client)
        end
    end

    return (client, device)
end

function compile_xla(
    f,
    args;
    before_xla_optimizations::Bool=false,
    client=nothing,
    serializable::Bool=false,
    kwargs...,
)
    # register MLIR dialects
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    context_gc_vector[ctx] = Vector{Union{TracedRArray,TracedRNumber}}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    client = client !== nothing ? client : XLA.default_backend()
    backend = XLA.platform_name(client)

    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    MLIR.IR.activate!(ctx)
    results = try
        # compile function to MLIR module
        mod = MLIR.IR.Module(MLIR.IR.Location())

        compile_options, kwargs = __get_compile_options_and_kwargs(; kwargs...)
        mlir_fn_res = compile_mlir!(
            mod,
            f,
            args,
            compile_options;
            backend,
            runtime=XLA.runtime(client),
            client,
            kwargs...,
        )

        # Resolve client and device
        client, device = __resolve_device_and_client(
            client,
            mlir_fn_res.seen_args,
            mlir_fn_res.linear_args,
            mlir_fn_res.is_sharded,
        )

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        # Drop some of our attributes
        run_pass_pipeline!(mod, "drop-unsupported-attributes", "drop_enzymexla_attributes")

        # compile MLIR module to XLA executable
        global_device_ids = collect(Int64, mlir_fn_res.global_device_ids)
        mlir_fn_res.is_sharded && (device = nothing)

        # XLA.compile mutates the module, for serialization we need to keep a copy
        if serializable
            iobuffer = IOBuffer()
            show(IOContext(iobuffer, :debug => true), mod)
            module_string = String(take!(iobuffer))
        else
            module_string = ""
        end

        if before_xla_optimizations
            exec = nothing
            hlo_modules = XLA.HloModule(mod)
        else
            exec = XLA.compile(
                client,
                device,
                mod;
                num_outputs=length(mlir_fn_res.linear_results),
                num_parameters=length(mlir_fn_res.linear_args),
                mlir_fn_res.is_sharded,
                global_device_ids,
                mlir_fn_res.num_replicas,
                mlir_fn_res.num_partitions,
                mlir_fn_res.use_shardy_partitioner,
            )
            hlo_modules = XLA.get_hlo_modules(exec)
            hlo_modules = length(hlo_modules) == 1 ? only(hlo_modules) : hlo_modules
        end

        return mod, exec, hlo_modules, mlir_fn_res, device, client, module_string
    finally
        MLIR.IR.deactivate!(ctx)
    end

    Base.delete!(context_gc_vector, ctx)
    return results
end

# inspired by RuntimeGeneratedFunction.jl
const __thunk_fwd_body_cache = Dict{Symbol,Expr}()
const __thunk_rev_body_cache = Dict{Expr,Symbol}()

function compile(f, args; kwargs...)
    compile_options, kwargs = __get_compile_options_and_kwargs(; kwargs...)

    _, exec, _, mlir_fn_res, device, client, str = compile_xla(
        f, args; compile_options, kwargs...
    )
    (;
        linear_args,
        seen_args,
        linear_results,
        preserved_args,
        concrete_result,
        donated_args_mask,
    ) = mlir_fn_res

    result_stores = Dict{Tuple,Symbol}()
    path_to_shard_info = mlir_fn_res.is_sharded ? Dict{Tuple,Symbol}() : nothing

    global_mesh_expr = if mlir_fn_res.unique_meshes === nothing
        :()
    elseif length(mlir_fn_res.unique_meshes) == 1
        only_mesh = only(mlir_fn_res.unique_meshes)
        :(Sharding.Mesh(
            thunk.global_device_ids, # same as only_mesh.global_device_ids
            0:(length(thunk.global_device_ids) - 1), # same as only_mesh.logical_device_ids
            $(only_mesh.axis_names),
            $(only_mesh.axis_sizes),
        ))
    else
        :(Sharding.Mesh(
            thunk.global_device_ids,
            0:(length(thunk.global_device_ids) - 1),
            (:flat_mesh,),
            (length(thunk.global_device_ids),),
        ))
    end

    ndevices = mlir_fn_res.is_sharded ? length(mlir_fn_res.global_device_ids) : 1

    # generate Julia `Thunk` code
    flatten_arg_names, flatten_code, resharded_inputs = codegen_flatten!(
        linear_args,
        seen_args,
        mlir_fn_res.is_sharded,
        XLA.get_parameter_shardings(exec), # TODO: use the same workflow as output shardings to parse the tensor sharding attributes directly if possible
        client,
        ndevices,
    )

    concretized_res_names, xla_call_code = codegen_xla_call(
        flatten_arg_names,
        length(linear_results),
        mlir_fn_res.is_sharded,
        ndevices,
        mlir_fn_res.is_pure,
    )

    shard_info_code, optional_shard_info_code, linear_result_shard_info = codegen_shard_info(
        mlir_fn_res.is_sharded,
        length(linear_results),
        linear_results,
        mlir_fn_res.result_shardings,
        exec,
        ndevices,
    )

    unflatten_code, used_shardinfo = codegen_unflatten!(
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

    for (i, name) in enumerate(linear_result_shard_info)
        if name in used_shardinfo
            push!(shard_info_code, optional_shard_info_code[i])
        end
    end

    sync_call = if compile_options.sync
        calls = []
        for name in concretized_res_names
            push!(calls, :(XLA.synced_buffer($(name))))
        end
        Expr(:block, calls...)
    else
        :()
    end

    donated_buffers_set = if XLA.runtime(client) isa Val{:PJRT}
        :(Base.IdSet{NTuple{<:Any,XLA.PJRT.Buffer}}())
    else
        :(Base.IdSet{XLA.IFRT.Array}())
    end

    body = quote
        global_mesh = $(global_mesh_expr)
        donated_buffers = $(donated_buffers_set)
        donated_args_mask = thunk.donated_args_mask
        $(flatten_code...)
        $(xla_call_code)
        $(sync_call)
        $(shard_info_code...)
        $(unflatten_code...)
        return result
    end

    if DEBUG_PRINT_CODEGEN[] && Reactant.Distributed.local_rank() == 0
        display(body)
        display(mlir_fn_res.donated_args_mask)
    end

    fname = if body in keys(__thunk_rev_body_cache)
        __thunk_rev_body_cache[body]
    else
        fname2 = gensym(Symbol(Symbol(f), :_reactant))
        __thunk_rev_body_cache[body] = fname2
        __thunk_fwd_body_cache[fname2] = body
        fname2
    end

    return register_thunk(
        fname,
        Tuple{map(Core.Typeof, args)...},
        f,
        mlir_fn_res.fnwrapped,
        exec,
        mlir_fn_res.is_sharded ? nothing : device,
        str,
        client,
        mlir_fn_res.global_device_ids,
        mlir_fn_res.donated_args_mask,
        compile_options.sync,
    )
end

struct Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD,DAM}
    f::FTy
    exec::ExecTy
    device::DeviceTy
    module_string::String
    client::ClientTy
    global_device_ids::GD
    donated_args_mask::DAM
    compiled_with_sync::Bool
end

for fn in (:get_tag, :get_isclosure, :get_compiled_argtypes)
    @eval $fn(thunk::Thunk) = $fn(typeof(thunk))
end

function get_compiled_argtypes(::Type{<:Thunk{<:Any,<:Any,<:Any,ArgTypes}}) where {ArgTypes}
    return ArgTypes
end

get_tag(::Type{<:Thunk{<:Any,tag}}) where {tag} = tag

get_isclosure(::Type{<:Thunk{<:Any,<:Any,IsClosure}}) where {IsClosure} = IsClosure

function Base.show(io::IO, thunk::Thunk{<:Any,tag}) where {tag}
    return print(io, "Reactant compiled function $(thunk.f) (with tag $(tag))")
end

XLA.cost_analysis(thunk::Thunk) = XLA.cost_analysis(thunk.exec)

XLA.get_output_shardings(thunk::Thunk) = XLA.get_output_shardings(thunk.exec)

XLA.get_parameter_shardings(thunk::Thunk) = XLA.get_parameter_shardings(thunk.exec)

struct MisMatchedThunkTypeError{ThunkTy,FoundTypes} <: Base.Exception end

function Base.showerror(
    io::IO,
    ::MisMatchedThunkTypeError{
        <:Thunk{FTy,tag,IsClosure,ArgTypes,ExecTy,DeviceTy,ClientTy,GD},FoundTypes
    },
) where {FTy,tag,ArgTypes,FoundTypes,IsClosure,ExecTy,DeviceTy,ClientTy,GD}
    print(
        io,
        "\nThe Reactant-compiled function \
         `$(Thunk{FTy, tag, ArgTypes, IsClosure, ExecTy, DeviceTy, ClientTy, GD})` exists, \
         but no method is defined for this combination of argument types.",
    )
    print(
        io,
        "\nYou passed in arguments with types\n\t(" *
        join(FoundTypes.parameters, ", ") *
        ")",
    )
    return print(
        io,
        "\nHowever the method you are calling was compiled for arguments with types\n\t(" *
        join(ArgTypes.parameters, ", ") *
        ")",
    )
end

@generated function (thunk::Thunk)(args...)
    FoundTypes = Tuple{args...}
    if get_compiled_argtypes(thunk) != FoundTypes
        return :(throw($(MisMatchedThunkTypeError{thunk,FoundTypes}())))
    end
    body = __thunk_fwd_body_cache[get_tag(thunk)]
    if get_isclosure(thunk)
        return quote
            args = (thunk.f, args...)
            $body
        end
    else
        return body
    end
end

function register_thunk(
    tag::Symbol,
    @nospecialize(argtys::Type),
    @nospecialize(f),
    isclosure::Bool,
    exec,
    device,
    module_string,
    client,
    global_device_ids,
    donated_args_mask,
    compiled_with_sync::Bool,
)
    return Thunk{
        Core.Typeof(f),
        tag,
        isclosure,
        argtys,
        Core.Typeof(exec),
        Core.Typeof(device),
        Core.Typeof(client),
        Core.Typeof(global_device_ids),
        Core.Typeof(donated_args_mask),
    }(
        f,
        exec,
        device,
        module_string,
        client,
        global_device_ids,
        donated_args_mask,
        compiled_with_sync,
    )
end

for cache_type in (:callcache, :sdycache, :sdygroupidcache)
    activate_fn = Symbol(:activate_, cache_type, :!)
    deactivate_fn = Symbol(:deactivate_, cache_type, :!)
    has_fn = Symbol(:_has_, cache_type)

    @eval begin
        function $(activate_fn)(cache)
            stack = get!(task_local_storage(), $(Meta.quot(cache_type))) do
                return []
            end
            push!(stack, cache)
            return nothing
        end

        function $(deactivate_fn)(cache)
            cache === last(task_local_storage($(Meta.quot(cache_type)))) ||
                error("Deactivating wrong cache")
            return pop!(task_local_storage($(Meta.quot(cache_type))))
        end

        function $(has_fn)()
            return haskey(task_local_storage(), $(Meta.quot(cache_type))) &&
                   !Base.isempty(task_local_storage($(Meta.quot(cache_type))))
        end

        function $(cache_type)(; throw_error::Bool=true)
            if !$(has_fn)()
                throw_error && error("No cache is active")
                return nothing
            end
            return last(task_local_storage($(Meta.quot(cache_type))))
        end
    end
end

function default_sdycache()
    return Dict{
        Tuple{AbstractVector{Int},NTuple{<:Any,Symbol},Dims{<:Any}},
        @NamedTuple{
            sym_name::MLIR.IR.Attribute,
            mesh_attr::MLIR.IR.Attribute,
            mesh_op::MLIR.IR.Operation,
            mesh::Sharding.Mesh,
        }
    }()
end

mutable struct SdyGroupIDCounter{T}
    @atomic group_id::T
end

function default_sdygroupidcache()
    return SdyGroupIDCounter{Int}(0), Base.IdDict{Union{TracedRArray,TracedRNumber},Int}()
end

function default_callcache()
    return Dict{
        Vector,
        @NamedTuple{
            f_name::String,
            mlir_result_types::Vector{MLIR.IR.Type},
            traced_result::Any,
            mutated_args::Vector{Int},
            linear_results::Vector{Reactant.TracedType},
            fnwrapped::Bool,
            argprefix::Symbol,
            resprefix::Symbol,
            resargprefix::Symbol,
        }
    }()
end

end
