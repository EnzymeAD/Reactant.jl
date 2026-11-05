import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export CublasLtMatmulAlgorithmProto


struct CublasLtMatmulAlgorithmProto
    algo_id::Int32
    tile_id::Int32
    split_k::Int32
    reduction_scheme::Int32
    cta_swizzling::Int32
    custom_option::Int32
    stages_id::Int32
    inner_shape_id::Int32
    cluster_shape_id::Int32
    numerical_impl_flags::Int64
end
PB.default_values(::Type{CublasLtMatmulAlgorithmProto}) = (;algo_id = zero(Int32), tile_id = zero(Int32), split_k = zero(Int32), reduction_scheme = zero(Int32), cta_swizzling = zero(Int32), custom_option = zero(Int32), stages_id = zero(Int32), inner_shape_id = zero(Int32), cluster_shape_id = zero(Int32), numerical_impl_flags = zero(Int64))
PB.field_numbers(::Type{CublasLtMatmulAlgorithmProto}) = (;algo_id = 1, tile_id = 2, split_k = 3, reduction_scheme = 4, cta_swizzling = 5, custom_option = 6, stages_id = 7, inner_shape_id = 8, cluster_shape_id = 9, numerical_impl_flags = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CublasLtMatmulAlgorithmProto}, _endpos::Int=0, _group::Bool=false)
    algo_id = zero(Int32)
    tile_id = zero(Int32)
    split_k = zero(Int32)
    reduction_scheme = zero(Int32)
    cta_swizzling = zero(Int32)
    custom_option = zero(Int32)
    stages_id = zero(Int32)
    inner_shape_id = zero(Int32)
    cluster_shape_id = zero(Int32)
    numerical_impl_flags = zero(Int64)
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            algo_id = PB.decode(d, Int32)
        elseif field_number == 2
            tile_id = PB.decode(d, Int32)
        elseif field_number == 3
            split_k = PB.decode(d, Int32)
        elseif field_number == 4
            reduction_scheme = PB.decode(d, Int32)
        elseif field_number == 5
            cta_swizzling = PB.decode(d, Int32)
        elseif field_number == 6
            custom_option = PB.decode(d, Int32)
        elseif field_number == 7
            stages_id = PB.decode(d, Int32)
        elseif field_number == 8
            inner_shape_id = PB.decode(d, Int32)
        elseif field_number == 9
            cluster_shape_id = PB.decode(d, Int32)
        elseif field_number == 10
            numerical_impl_flags = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return CublasLtMatmulAlgorithmProto(algo_id, tile_id, split_k, reduction_scheme, cta_swizzling, custom_option, stages_id, inner_shape_id, cluster_shape_id, numerical_impl_flags)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CublasLtMatmulAlgorithmProto)
    initpos = position(e.io)
    x.algo_id != zero(Int32) && PB.encode(e, 1, x.algo_id)
    x.tile_id != zero(Int32) && PB.encode(e, 2, x.tile_id)
    x.split_k != zero(Int32) && PB.encode(e, 3, x.split_k)
    x.reduction_scheme != zero(Int32) && PB.encode(e, 4, x.reduction_scheme)
    x.cta_swizzling != zero(Int32) && PB.encode(e, 5, x.cta_swizzling)
    x.custom_option != zero(Int32) && PB.encode(e, 6, x.custom_option)
    x.stages_id != zero(Int32) && PB.encode(e, 7, x.stages_id)
    x.inner_shape_id != zero(Int32) && PB.encode(e, 8, x.inner_shape_id)
    x.cluster_shape_id != zero(Int32) && PB.encode(e, 9, x.cluster_shape_id)
    x.numerical_impl_flags != zero(Int64) && PB.encode(e, 10, x.numerical_impl_flags)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CublasLtMatmulAlgorithmProto)
    encoded_size = 0
    x.algo_id != zero(Int32) && (encoded_size += PB._encoded_size(x.algo_id, 1))
    x.tile_id != zero(Int32) && (encoded_size += PB._encoded_size(x.tile_id, 2))
    x.split_k != zero(Int32) && (encoded_size += PB._encoded_size(x.split_k, 3))
    x.reduction_scheme != zero(Int32) && (encoded_size += PB._encoded_size(x.reduction_scheme, 4))
    x.cta_swizzling != zero(Int32) && (encoded_size += PB._encoded_size(x.cta_swizzling, 5))
    x.custom_option != zero(Int32) && (encoded_size += PB._encoded_size(x.custom_option, 6))
    x.stages_id != zero(Int32) && (encoded_size += PB._encoded_size(x.stages_id, 7))
    x.inner_shape_id != zero(Int32) && (encoded_size += PB._encoded_size(x.inner_shape_id, 8))
    x.cluster_shape_id != zero(Int32) && (encoded_size += PB._encoded_size(x.cluster_shape_id, 9))
    x.numerical_impl_flags != zero(Int64) && (encoded_size += PB._encoded_size(x.numerical_impl_flags, 10))
    return encoded_size
end
