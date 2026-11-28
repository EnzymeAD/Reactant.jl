import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export BlasLtMatmulAlgorithmProto


struct BlasLtMatmulAlgorithmProto
    library_version::Int64
    algorithm::Union{Nothing,OneOf{stream_executor_cuda.CublasLtMatmulAlgorithmProto}}
end
PB.oneof_field_types(::Type{BlasLtMatmulAlgorithmProto}) = (;
    algorithm = (;cublaslt=stream_executor_cuda.CublasLtMatmulAlgorithmProto),
)
PB.default_values(::Type{BlasLtMatmulAlgorithmProto}) = (;library_version = zero(Int64), cublaslt = nothing)
PB.field_numbers(::Type{BlasLtMatmulAlgorithmProto}) = (;library_version = 1, cublaslt = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BlasLtMatmulAlgorithmProto}, _endpos::Int=0, _group::Bool=false)
    library_version = zero(Int64)
    algorithm = nothing
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            library_version = PB.decode(d, Int64)
        elseif field_number == 2
            algorithm = OneOf(:cublaslt, PB.decode(d, Ref{stream_executor_cuda.CublasLtMatmulAlgorithmProto}))
        else
            Base.skip(d, wire_type)
        end
    end
    return BlasLtMatmulAlgorithmProto(library_version, algorithm)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BlasLtMatmulAlgorithmProto)
    initpos = position(e.io)
    x.library_version != zero(Int64) && PB.encode(e, 1, x.library_version)
    if isnothing(x.algorithm);
    elseif x.algorithm.name === :cublaslt
        PB.encode(e, 2, x.algorithm[]::stream_executor_cuda.CublasLtMatmulAlgorithmProto)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::BlasLtMatmulAlgorithmProto)
    encoded_size = 0
    x.library_version != zero(Int64) && (encoded_size += PB._encoded_size(x.library_version, 1))
    if isnothing(x.algorithm);
    elseif x.algorithm.name === :cublaslt
        encoded_size += PB._encoded_size(x.algorithm[]::stream_executor_cuda.CublasLtMatmulAlgorithmProto, 2)
    end
    return encoded_size
end
