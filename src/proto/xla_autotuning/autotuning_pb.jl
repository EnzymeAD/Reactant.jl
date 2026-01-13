import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"AutotuneResult.TritonGemmKey", var"AutotuneResult.BackendConfigKey"
export var"AutotuneResult.ConvKey", CudnnVersion, var"AutotuneResult.FailureKind"
export var"AutotuneResult.GemmKey", var"AutotuneResult.CustomKernelFusionKey"
export var"AutotuneResult.CudaConvPlanKey", ComputeCapability, TritonGemmConfigsProto
export var"AutotuneResult.FailureResult", AutotuneResult, AutotuningLog


struct var"AutotuneResult.TritonGemmKey"
    block_m::Int64
    block_n::Int64
    block_k::Int64
    split_k::Int64
    num_stages::Int64
    num_warps::Int64
    num_ctas::Int64
    is_tma_allowed::Bool
    is_warp_specialization_allowed::Bool
end
var"AutotuneResult.TritonGemmKey"(;block_m = zero(Int64), block_n = zero(Int64), block_k = zero(Int64), split_k = zero(Int64), num_stages = zero(Int64), num_warps = zero(Int64), num_ctas = zero(Int64), is_tma_allowed = false, is_warp_specialization_allowed = false) = var"AutotuneResult.TritonGemmKey"(block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas, is_tma_allowed, is_warp_specialization_allowed)
PB.default_values(::Type{var"AutotuneResult.TritonGemmKey"}) = (;block_m = zero(Int64), block_n = zero(Int64), block_k = zero(Int64), split_k = zero(Int64), num_stages = zero(Int64), num_warps = zero(Int64), num_ctas = zero(Int64), is_tma_allowed = false, is_warp_specialization_allowed = false)
PB.field_numbers(::Type{var"AutotuneResult.TritonGemmKey"}) = (;block_m = 1, block_n = 2, block_k = 3, split_k = 4, num_stages = 5, num_warps = 6, num_ctas = 7, is_tma_allowed = 8, is_warp_specialization_allowed = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.TritonGemmKey"})
    block_m = zero(Int64)
    block_n = zero(Int64)
    block_k = zero(Int64)
    split_k = zero(Int64)
    num_stages = zero(Int64)
    num_warps = zero(Int64)
    num_ctas = zero(Int64)
    is_tma_allowed = false
    is_warp_specialization_allowed = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            block_m = PB.decode(d, Int64)
        elseif field_number == 2
            block_n = PB.decode(d, Int64)
        elseif field_number == 3
            block_k = PB.decode(d, Int64)
        elseif field_number == 4
            split_k = PB.decode(d, Int64)
        elseif field_number == 5
            num_stages = PB.decode(d, Int64)
        elseif field_number == 6
            num_warps = PB.decode(d, Int64)
        elseif field_number == 7
            num_ctas = PB.decode(d, Int64)
        elseif field_number == 8
            is_tma_allowed = PB.decode(d, Bool)
        elseif field_number == 9
            is_warp_specialization_allowed = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.TritonGemmKey"(block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas, is_tma_allowed, is_warp_specialization_allowed)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.TritonGemmKey")
    initpos = position(e.io)
    x.block_m != zero(Int64) && PB.encode(e, 1, x.block_m)
    x.block_n != zero(Int64) && PB.encode(e, 2, x.block_n)
    x.block_k != zero(Int64) && PB.encode(e, 3, x.block_k)
    x.split_k != zero(Int64) && PB.encode(e, 4, x.split_k)
    x.num_stages != zero(Int64) && PB.encode(e, 5, x.num_stages)
    x.num_warps != zero(Int64) && PB.encode(e, 6, x.num_warps)
    x.num_ctas != zero(Int64) && PB.encode(e, 7, x.num_ctas)
    x.is_tma_allowed != false && PB.encode(e, 8, x.is_tma_allowed)
    x.is_warp_specialization_allowed != false && PB.encode(e, 9, x.is_warp_specialization_allowed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.TritonGemmKey")
    encoded_size = 0
    x.block_m != zero(Int64) && (encoded_size += PB._encoded_size(x.block_m, 1))
    x.block_n != zero(Int64) && (encoded_size += PB._encoded_size(x.block_n, 2))
    x.block_k != zero(Int64) && (encoded_size += PB._encoded_size(x.block_k, 3))
    x.split_k != zero(Int64) && (encoded_size += PB._encoded_size(x.split_k, 4))
    x.num_stages != zero(Int64) && (encoded_size += PB._encoded_size(x.num_stages, 5))
    x.num_warps != zero(Int64) && (encoded_size += PB._encoded_size(x.num_warps, 6))
    x.num_ctas != zero(Int64) && (encoded_size += PB._encoded_size(x.num_ctas, 7))
    x.is_tma_allowed != false && (encoded_size += PB._encoded_size(x.is_tma_allowed, 8))
    x.is_warp_specialization_allowed != false && (encoded_size += PB._encoded_size(x.is_warp_specialization_allowed, 9))
    return encoded_size
end

struct var"AutotuneResult.BackendConfigKey"
    name::String
    config::Union{Nothing,google.protobuf.var"#Any"}
end
var"AutotuneResult.BackendConfigKey"(;name = "", config = nothing) = var"AutotuneResult.BackendConfigKey"(name, config)
PB.default_values(::Type{var"AutotuneResult.BackendConfigKey"}) = (;name = "", config = nothing)
PB.field_numbers(::Type{var"AutotuneResult.BackendConfigKey"}) = (;name = 1, config = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.BackendConfigKey"})
    name = ""
    config = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, config)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.BackendConfigKey"(name, config[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.BackendConfigKey")
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isnothing(x.config) && PB.encode(e, 2, x.config)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.BackendConfigKey")
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isnothing(x.config) && (encoded_size += PB._encoded_size(x.config, 2))
    return encoded_size
end

struct var"AutotuneResult.ConvKey"
    algorithm::Int64
    tensor_ops_enabled::Bool
end
var"AutotuneResult.ConvKey"(;algorithm = zero(Int64), tensor_ops_enabled = false) = var"AutotuneResult.ConvKey"(algorithm, tensor_ops_enabled)
PB.default_values(::Type{var"AutotuneResult.ConvKey"}) = (;algorithm = zero(Int64), tensor_ops_enabled = false)
PB.field_numbers(::Type{var"AutotuneResult.ConvKey"}) = (;algorithm = 1, tensor_ops_enabled = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.ConvKey"})
    algorithm = zero(Int64)
    tensor_ops_enabled = false
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            algorithm = PB.decode(d, Int64)
        elseif field_number == 2
            tensor_ops_enabled = PB.decode(d, Bool)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.ConvKey"(algorithm, tensor_ops_enabled)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.ConvKey")
    initpos = position(e.io)
    x.algorithm != zero(Int64) && PB.encode(e, 1, x.algorithm)
    x.tensor_ops_enabled != false && PB.encode(e, 2, x.tensor_ops_enabled)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.ConvKey")
    encoded_size = 0
    x.algorithm != zero(Int64) && (encoded_size += PB._encoded_size(x.algorithm, 1))
    x.tensor_ops_enabled != false && (encoded_size += PB._encoded_size(x.tensor_ops_enabled, 2))
    return encoded_size
end

struct CudnnVersion
    major::Int32
    minor::Int32
    patch::Int32
end
CudnnVersion(;major = zero(Int32), minor = zero(Int32), patch = zero(Int32)) = CudnnVersion(major, minor, patch)
PB.default_values(::Type{CudnnVersion}) = (;major = zero(Int32), minor = zero(Int32), patch = zero(Int32))
PB.field_numbers(::Type{CudnnVersion}) = (;major = 1, minor = 2, patch = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CudnnVersion})
    major = zero(Int32)
    minor = zero(Int32)
    patch = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, Int32)
        elseif field_number == 2
            minor = PB.decode(d, Int32)
        elseif field_number == 3
            patch = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return CudnnVersion(major, minor, patch)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CudnnVersion)
    initpos = position(e.io)
    x.major != zero(Int32) && PB.encode(e, 1, x.major)
    x.minor != zero(Int32) && PB.encode(e, 2, x.minor)
    x.patch != zero(Int32) && PB.encode(e, 3, x.patch)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CudnnVersion)
    encoded_size = 0
    x.major != zero(Int32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(Int32) && (encoded_size += PB._encoded_size(x.minor, 2))
    x.patch != zero(Int32) && (encoded_size += PB._encoded_size(x.patch, 3))
    return encoded_size
end

@enumx var"AutotuneResult.FailureKind" UNKNOWN=0 REDZONE_MODIFIED=1 WRONG_RESULT=2 DISQUALIFIED=3

struct var"AutotuneResult.GemmKey"
    algorithm::Int64
end
var"AutotuneResult.GemmKey"(;algorithm = zero(Int64)) = var"AutotuneResult.GemmKey"(algorithm)
PB.default_values(::Type{var"AutotuneResult.GemmKey"}) = (;algorithm = zero(Int64))
PB.field_numbers(::Type{var"AutotuneResult.GemmKey"}) = (;algorithm = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.GemmKey"})
    algorithm = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            algorithm = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.GemmKey"(algorithm)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.GemmKey")
    initpos = position(e.io)
    x.algorithm != zero(Int64) && PB.encode(e, 1, x.algorithm)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.GemmKey")
    encoded_size = 0
    x.algorithm != zero(Int64) && (encoded_size += PB._encoded_size(x.algorithm, 1))
    return encoded_size
end

struct var"AutotuneResult.CustomKernelFusionKey"
    kernel_index::Int64
end
var"AutotuneResult.CustomKernelFusionKey"(;kernel_index = zero(Int64)) = var"AutotuneResult.CustomKernelFusionKey"(kernel_index)
PB.default_values(::Type{var"AutotuneResult.CustomKernelFusionKey"}) = (;kernel_index = zero(Int64))
PB.field_numbers(::Type{var"AutotuneResult.CustomKernelFusionKey"}) = (;kernel_index = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.CustomKernelFusionKey"})
    kernel_index = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kernel_index = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.CustomKernelFusionKey"(kernel_index)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.CustomKernelFusionKey")
    initpos = position(e.io)
    x.kernel_index != zero(Int64) && PB.encode(e, 1, x.kernel_index)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.CustomKernelFusionKey")
    encoded_size = 0
    x.kernel_index != zero(Int64) && (encoded_size += PB._encoded_size(x.kernel_index, 1))
    return encoded_size
end

struct var"AutotuneResult.CudaConvPlanKey"
    exec_plan_id::String
end
var"AutotuneResult.CudaConvPlanKey"(;exec_plan_id = "") = var"AutotuneResult.CudaConvPlanKey"(exec_plan_id)
PB.default_values(::Type{var"AutotuneResult.CudaConvPlanKey"}) = (;exec_plan_id = "")
PB.field_numbers(::Type{var"AutotuneResult.CudaConvPlanKey"}) = (;exec_plan_id = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.CudaConvPlanKey"})
    exec_plan_id = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            exec_plan_id = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.CudaConvPlanKey"(exec_plan_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.CudaConvPlanKey")
    initpos = position(e.io)
    !isempty(x.exec_plan_id) && PB.encode(e, 1, x.exec_plan_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.CudaConvPlanKey")
    encoded_size = 0
    !isempty(x.exec_plan_id) && (encoded_size += PB._encoded_size(x.exec_plan_id, 1))
    return encoded_size
end

struct ComputeCapability
    major::Int32
    minor::Int32
end
ComputeCapability(;major = zero(Int32), minor = zero(Int32)) = ComputeCapability(major, minor)
PB.default_values(::Type{ComputeCapability}) = (;major = zero(Int32), minor = zero(Int32))
PB.field_numbers(::Type{ComputeCapability}) = (;major = 1, minor = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ComputeCapability})
    major = zero(Int32)
    minor = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            major = PB.decode(d, Int32)
        elseif field_number == 2
            minor = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return ComputeCapability(major, minor)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ComputeCapability)
    initpos = position(e.io)
    x.major != zero(Int32) && PB.encode(e, 1, x.major)
    x.minor != zero(Int32) && PB.encode(e, 2, x.minor)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ComputeCapability)
    encoded_size = 0
    x.major != zero(Int32) && (encoded_size += PB._encoded_size(x.major, 1))
    x.minor != zero(Int32) && (encoded_size += PB._encoded_size(x.minor, 2))
    return encoded_size
end

struct TritonGemmConfigsProto
    config::Vector{var"AutotuneResult.TritonGemmKey"}
end
TritonGemmConfigsProto(;config = Vector{var"AutotuneResult.TritonGemmKey"}()) = TritonGemmConfigsProto(config)
PB.default_values(::Type{TritonGemmConfigsProto}) = (;config = Vector{var"AutotuneResult.TritonGemmKey"}())
PB.field_numbers(::Type{TritonGemmConfigsProto}) = (;config = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TritonGemmConfigsProto})
    config = PB.BufferedVector{var"AutotuneResult.TritonGemmKey"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, config)
        else
            Base.skip(d, wire_type)
        end
    end
    return TritonGemmConfigsProto(config[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TritonGemmConfigsProto)
    initpos = position(e.io)
    !isempty(x.config) && PB.encode(e, 1, x.config)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TritonGemmConfigsProto)
    encoded_size = 0
    !isempty(x.config) && (encoded_size += PB._encoded_size(x.config, 1))
    return encoded_size
end

struct var"AutotuneResult.FailureResult"
    kind::var"AutotuneResult.FailureKind".T
    msg::String
    key::Union{Nothing,OneOf{<:Union{var"AutotuneResult.ConvKey",var"AutotuneResult.GemmKey",var"AutotuneResult.CudaConvPlanKey",xla_tsl_dnn.AlgorithmProto}}}
    buffer_address::Int64
end
var"AutotuneResult.FailureResult"(;kind = var"AutotuneResult.FailureKind".UNKNOWN, msg = "", key = nothing, buffer_address = zero(Int64)) = var"AutotuneResult.FailureResult"(kind, msg, key, buffer_address)
PB.oneof_field_types(::Type{var"AutotuneResult.FailureResult"}) = (;
    key = (;reference_conv=var"AutotuneResult.ConvKey", reference_gemm=var"AutotuneResult.GemmKey", reference_cuda_conv_plan=var"AutotuneResult.CudaConvPlanKey", reference_algorithm=xla_tsl_dnn.AlgorithmProto),
)
PB.default_values(::Type{var"AutotuneResult.FailureResult"}) = (;kind = var"AutotuneResult.FailureKind".UNKNOWN, msg = "", reference_conv = nothing, reference_gemm = nothing, reference_cuda_conv_plan = nothing, reference_algorithm = nothing, buffer_address = zero(Int64))
PB.field_numbers(::Type{var"AutotuneResult.FailureResult"}) = (;kind = 1, msg = 2, reference_conv = 11, reference_gemm = 12, reference_cuda_conv_plan = 14, reference_algorithm = 15, buffer_address = 13)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AutotuneResult.FailureResult"})
    kind = var"AutotuneResult.FailureKind".UNKNOWN
    msg = ""
    key = nothing
    buffer_address = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kind = PB.decode(d, var"AutotuneResult.FailureKind".T)
        elseif field_number == 2
            msg = PB.decode(d, String)
        elseif field_number == 11
            key = OneOf(:reference_conv, PB.decode(d, Ref{var"AutotuneResult.ConvKey"}))
        elseif field_number == 12
            key = OneOf(:reference_gemm, PB.decode(d, Ref{var"AutotuneResult.GemmKey"}))
        elseif field_number == 14
            key = OneOf(:reference_cuda_conv_plan, PB.decode(d, Ref{var"AutotuneResult.CudaConvPlanKey"}))
        elseif field_number == 15
            key = OneOf(:reference_algorithm, PB.decode(d, Ref{xla_tsl_dnn.AlgorithmProto}))
        elseif field_number == 13
            buffer_address = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"AutotuneResult.FailureResult"(kind, msg, key, buffer_address)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AutotuneResult.FailureResult")
    initpos = position(e.io)
    x.kind != var"AutotuneResult.FailureKind".UNKNOWN && PB.encode(e, 1, x.kind)
    !isempty(x.msg) && PB.encode(e, 2, x.msg)
    if isnothing(x.key);
    elseif x.key.name === :reference_conv
        PB.encode(e, 11, x.key[]::var"AutotuneResult.ConvKey")
    elseif x.key.name === :reference_gemm
        PB.encode(e, 12, x.key[]::var"AutotuneResult.GemmKey")
    elseif x.key.name === :reference_cuda_conv_plan
        PB.encode(e, 14, x.key[]::var"AutotuneResult.CudaConvPlanKey")
    elseif x.key.name === :reference_algorithm
        PB.encode(e, 15, x.key[]::xla_tsl_dnn.AlgorithmProto)
    end
    x.buffer_address != zero(Int64) && PB.encode(e, 13, x.buffer_address)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AutotuneResult.FailureResult")
    encoded_size = 0
    x.kind != var"AutotuneResult.FailureKind".UNKNOWN && (encoded_size += PB._encoded_size(x.kind, 1))
    !isempty(x.msg) && (encoded_size += PB._encoded_size(x.msg, 2))
    if isnothing(x.key);
    elseif x.key.name === :reference_conv
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.ConvKey", 11)
    elseif x.key.name === :reference_gemm
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.GemmKey", 12)
    elseif x.key.name === :reference_cuda_conv_plan
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.CudaConvPlanKey", 14)
    elseif x.key.name === :reference_algorithm
        encoded_size += PB._encoded_size(x.key[]::xla_tsl_dnn.AlgorithmProto, 15)
    end
    x.buffer_address != zero(Int64) && (encoded_size += PB._encoded_size(x.buffer_address, 13))
    return encoded_size
end

struct AutotuneResult
    scratch_bytes::Int64
    run_time::Union{Nothing,google.protobuf.Duration}
    failure::Union{Nothing,var"AutotuneResult.FailureResult"}
    key::Union{Nothing,OneOf{<:Union{var"AutotuneResult.ConvKey",var"AutotuneResult.GemmKey",var"AutotuneResult.TritonGemmKey",var"AutotuneResult.CudaConvPlanKey",var"AutotuneResult.CustomKernelFusionKey",xla_tsl_dnn.AlgorithmProto,var"AutotuneResult.BackendConfigKey"}}}
end
AutotuneResult(;scratch_bytes = zero(Int64), run_time = nothing, failure = nothing, key = nothing) = AutotuneResult(scratch_bytes, run_time, failure, key)
PB.oneof_field_types(::Type{AutotuneResult}) = (;
    key = (;conv=var"AutotuneResult.ConvKey", gemm=var"AutotuneResult.GemmKey", triton=var"AutotuneResult.TritonGemmKey", cuda_conv_plan=var"AutotuneResult.CudaConvPlanKey", custom_kernel_fusion=var"AutotuneResult.CustomKernelFusionKey", algorithm=xla_tsl_dnn.AlgorithmProto, other=var"AutotuneResult.BackendConfigKey"),
)
PB.default_values(::Type{AutotuneResult}) = (;scratch_bytes = zero(Int64), run_time = nothing, failure = nothing, conv = nothing, gemm = nothing, triton = nothing, cuda_conv_plan = nothing, custom_kernel_fusion = nothing, algorithm = nothing, other = nothing)
PB.field_numbers(::Type{AutotuneResult}) = (;scratch_bytes = 8, run_time = 9, failure = 7, conv = 5, gemm = 6, triton = 17, cuda_conv_plan = 15, custom_kernel_fusion = 18, algorithm = 16, other = 19)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AutotuneResult})
    scratch_bytes = zero(Int64)
    run_time = Ref{Union{Nothing,google.protobuf.Duration}}(nothing)
    failure = Ref{Union{Nothing,var"AutotuneResult.FailureResult"}}(nothing)
    key = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 8
            scratch_bytes = PB.decode(d, Int64)
        elseif field_number == 9
            PB.decode!(d, run_time)
        elseif field_number == 7
            PB.decode!(d, failure)
        elseif field_number == 5
            key = OneOf(:conv, PB.decode(d, Ref{var"AutotuneResult.ConvKey"}))
        elseif field_number == 6
            key = OneOf(:gemm, PB.decode(d, Ref{var"AutotuneResult.GemmKey"}))
        elseif field_number == 17
            key = OneOf(:triton, PB.decode(d, Ref{var"AutotuneResult.TritonGemmKey"}))
        elseif field_number == 15
            key = OneOf(:cuda_conv_plan, PB.decode(d, Ref{var"AutotuneResult.CudaConvPlanKey"}))
        elseif field_number == 18
            key = OneOf(:custom_kernel_fusion, PB.decode(d, Ref{var"AutotuneResult.CustomKernelFusionKey"}))
        elseif field_number == 16
            key = OneOf(:algorithm, PB.decode(d, Ref{xla_tsl_dnn.AlgorithmProto}))
        elseif field_number == 19
            key = OneOf(:other, PB.decode(d, Ref{var"AutotuneResult.BackendConfigKey"}))
        else
            Base.skip(d, wire_type)
        end
    end
    return AutotuneResult(scratch_bytes, run_time[], failure[], key)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AutotuneResult)
    initpos = position(e.io)
    x.scratch_bytes != zero(Int64) && PB.encode(e, 8, x.scratch_bytes)
    !isnothing(x.run_time) && PB.encode(e, 9, x.run_time)
    !isnothing(x.failure) && PB.encode(e, 7, x.failure)
    if isnothing(x.key);
    elseif x.key.name === :conv
        PB.encode(e, 5, x.key[]::var"AutotuneResult.ConvKey")
    elseif x.key.name === :gemm
        PB.encode(e, 6, x.key[]::var"AutotuneResult.GemmKey")
    elseif x.key.name === :triton
        PB.encode(e, 17, x.key[]::var"AutotuneResult.TritonGemmKey")
    elseif x.key.name === :cuda_conv_plan
        PB.encode(e, 15, x.key[]::var"AutotuneResult.CudaConvPlanKey")
    elseif x.key.name === :custom_kernel_fusion
        PB.encode(e, 18, x.key[]::var"AutotuneResult.CustomKernelFusionKey")
    elseif x.key.name === :algorithm
        PB.encode(e, 16, x.key[]::xla_tsl_dnn.AlgorithmProto)
    elseif x.key.name === :other
        PB.encode(e, 19, x.key[]::var"AutotuneResult.BackendConfigKey")
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::AutotuneResult)
    encoded_size = 0
    x.scratch_bytes != zero(Int64) && (encoded_size += PB._encoded_size(x.scratch_bytes, 8))
    !isnothing(x.run_time) && (encoded_size += PB._encoded_size(x.run_time, 9))
    !isnothing(x.failure) && (encoded_size += PB._encoded_size(x.failure, 7))
    if isnothing(x.key);
    elseif x.key.name === :conv
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.ConvKey", 5)
    elseif x.key.name === :gemm
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.GemmKey", 6)
    elseif x.key.name === :triton
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.TritonGemmKey", 17)
    elseif x.key.name === :cuda_conv_plan
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.CudaConvPlanKey", 15)
    elseif x.key.name === :custom_kernel_fusion
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.CustomKernelFusionKey", 18)
    elseif x.key.name === :algorithm
        encoded_size += PB._encoded_size(x.key[]::xla_tsl_dnn.AlgorithmProto, 16)
    elseif x.key.name === :other
        encoded_size += PB._encoded_size(x.key[]::var"AutotuneResult.BackendConfigKey", 19)
    end
    return encoded_size
end

struct AutotuningLog
    instr::Union{Nothing,google.protobuf.var"#Any"}
    results::Vector{AutotuneResult}
    cudnn_version::Union{Nothing,CudnnVersion}
    compute_capability::Union{Nothing,ComputeCapability}
    device_pci_bus_id::String
    blas_version::String
    fusion_name::String
    fusion_count::Int64
    selected_backend::String
end
AutotuningLog(;instr = nothing, results = Vector{AutotuneResult}(), cudnn_version = nothing, compute_capability = nothing, device_pci_bus_id = "", blas_version = "", fusion_name = "", fusion_count = zero(Int64), selected_backend = "") = AutotuningLog(instr, results, cudnn_version, compute_capability, device_pci_bus_id, blas_version, fusion_name, fusion_count, selected_backend)
PB.default_values(::Type{AutotuningLog}) = (;instr = nothing, results = Vector{AutotuneResult}(), cudnn_version = nothing, compute_capability = nothing, device_pci_bus_id = "", blas_version = "", fusion_name = "", fusion_count = zero(Int64), selected_backend = "")
PB.field_numbers(::Type{AutotuningLog}) = (;instr = 1, results = 2, cudnn_version = 3, compute_capability = 4, device_pci_bus_id = 5, blas_version = 6, fusion_name = 7, fusion_count = 8, selected_backend = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AutotuningLog})
    instr = Ref{Union{Nothing,google.protobuf.var"#Any"}}(nothing)
    results = PB.BufferedVector{AutotuneResult}()
    cudnn_version = Ref{Union{Nothing,CudnnVersion}}(nothing)
    compute_capability = Ref{Union{Nothing,ComputeCapability}}(nothing)
    device_pci_bus_id = ""
    blas_version = ""
    fusion_name = ""
    fusion_count = zero(Int64)
    selected_backend = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, instr)
        elseif field_number == 2
            PB.decode!(d, results)
        elseif field_number == 3
            PB.decode!(d, cudnn_version)
        elseif field_number == 4
            PB.decode!(d, compute_capability)
        elseif field_number == 5
            device_pci_bus_id = PB.decode(d, String)
        elseif field_number == 6
            blas_version = PB.decode(d, String)
        elseif field_number == 7
            fusion_name = PB.decode(d, String)
        elseif field_number == 8
            fusion_count = PB.decode(d, Int64)
        elseif field_number == 9
            selected_backend = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return AutotuningLog(instr[], results[], cudnn_version[], compute_capability[], device_pci_bus_id, blas_version, fusion_name, fusion_count, selected_backend)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AutotuningLog)
    initpos = position(e.io)
    !isnothing(x.instr) && PB.encode(e, 1, x.instr)
    !isempty(x.results) && PB.encode(e, 2, x.results)
    !isnothing(x.cudnn_version) && PB.encode(e, 3, x.cudnn_version)
    !isnothing(x.compute_capability) && PB.encode(e, 4, x.compute_capability)
    !isempty(x.device_pci_bus_id) && PB.encode(e, 5, x.device_pci_bus_id)
    !isempty(x.blas_version) && PB.encode(e, 6, x.blas_version)
    !isempty(x.fusion_name) && PB.encode(e, 7, x.fusion_name)
    x.fusion_count != zero(Int64) && PB.encode(e, 8, x.fusion_count)
    !isempty(x.selected_backend) && PB.encode(e, 9, x.selected_backend)
    return position(e.io) - initpos
end
function PB._encoded_size(x::AutotuningLog)
    encoded_size = 0
    !isnothing(x.instr) && (encoded_size += PB._encoded_size(x.instr, 1))
    !isempty(x.results) && (encoded_size += PB._encoded_size(x.results, 2))
    !isnothing(x.cudnn_version) && (encoded_size += PB._encoded_size(x.cudnn_version, 3))
    !isnothing(x.compute_capability) && (encoded_size += PB._encoded_size(x.compute_capability, 4))
    !isempty(x.device_pci_bus_id) && (encoded_size += PB._encoded_size(x.device_pci_bus_id, 5))
    !isempty(x.blas_version) && (encoded_size += PB._encoded_size(x.blas_version, 6))
    !isempty(x.fusion_name) && (encoded_size += PB._encoded_size(x.fusion_name, 7))
    x.fusion_count != zero(Int64) && (encoded_size += PB._encoded_size(x.fusion_count, 8))
    !isempty(x.selected_backend) && (encoded_size += PB._encoded_size(x.selected_backend, 9))
    return encoded_size
end
