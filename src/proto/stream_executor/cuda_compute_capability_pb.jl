import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"CudaComputeCapabilityProto.FeatureExtension", CudaComputeCapabilityProto


@enumx var"CudaComputeCapabilityProto.FeatureExtension" UNSPECIFIED=0 NONE=1 ACCELERATED_FEATURES=2 FAMILY_COMPATIBLE_FEATURES=3

struct CudaComputeCapabilityProto
    major::Int32
    minor::Int32
    feature_extension::var"CudaComputeCapabilityProto.FeatureExtension".T
end
CudaComputeCapabilityProto(;major = zero(Int32), minor = zero(Int32), feature_extension = var"CudaComputeCapabilityProto.FeatureExtension".UNSPECIFIED) = CudaComputeCapabilityProto(major, minor, feature_extension)
PB.default_values(::Type{CudaComputeCapabilityProto}) = (;major = zero(Int32), minor = zero(Int32), feature_extension = var"CudaComputeCapabilityProto.FeatureExtension".UNSPECIFIED)
PB.field_numbers(::Type{CudaComputeCapabilityProto}) = (;major = 1, minor = 2, feature_extension = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CudaComputeCapabilityProto})
    major = zero(Int32)
    minor = zero(Int32)
    feature_extension = var"CudaComputeCapabilityProto.FeatureExtension".UNSPECIFIED
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, Int32)
        elseif field_number == 2
            minor = PB.decode(d, Int32)
        elseif field_number == 3
            feature_extension = PB.decode(d, var"CudaComputeCapabilityProto.FeatureExtension".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return CudaComputeCapabilityProto(major, minor, feature_extension)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CudaComputeCapabilityProto)
    initpos = position(e.io)
    x.major != zero(Int32) && PB.encode(e, 1, x.major)
    x.minor != zero(Int32) && PB.encode(e, 2, x.minor)
    x.feature_extension != var"CudaComputeCapabilityProto.FeatureExtension".UNSPECIFIED && PB.encode(e, 3, x.feature_extension)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CudaComputeCapabilityProto)
    encoded_size = 0
    x.major != zero(Int32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(Int32) && (encoded_size += PB._encoded_size(x.minor, 2))
    x.feature_extension != var"CudaComputeCapabilityProto.FeatureExtension".UNSPECIFIED && (encoded_size += PB._encoded_size(x.feature_extension, 3))
    return encoded_size
end
