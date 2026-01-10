import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"Node.InstructionCategory", var"Node.XLAInstruction.LayoutAnalysis.Dimension"
export Metrics, var"Node.XLAInstruction.LayoutAnalysis", var"Node.XLAInstruction", Node
export Profile


struct var"Node.InstructionCategory" end

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"Node.InstructionCategory"})
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        Base.skip(d, wire_type)
    end
    return var"Node.InstructionCategory"()
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"Node.InstructionCategory")
    initpos = position(e.io)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"Node.InstructionCategory")
    encoded_size = 0
    return encoded_size
end

struct var"Node.XLAInstruction.LayoutAnalysis.Dimension"
    size::Int32
    alignment::Int32
    semantics::String
end
var"Node.XLAInstruction.LayoutAnalysis.Dimension"(;size = zero(Int32), alignment = zero(Int32), semantics = "") = var"Node.XLAInstruction.LayoutAnalysis.Dimension"(size, alignment, semantics)
PB.default_values(::Type{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}) = (;size = zero(Int32), alignment = zero(Int32), semantics = "")
PB.field_numbers(::Type{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}) = (;size = 1, alignment = 2, semantics = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"Node.XLAInstruction.LayoutAnalysis.Dimension"})
    size = zero(Int32)
    alignment = zero(Int32)
    semantics = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            size = PB.decode(d, Int32)
        elseif field_number == 2
            alignment = PB.decode(d, Int32)
        elseif field_number == 3
            semantics = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"Node.XLAInstruction.LayoutAnalysis.Dimension"(size, alignment, semantics)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"Node.XLAInstruction.LayoutAnalysis.Dimension")
    initpos = position(e.io)
    x.size != zero(Int32) && PB.encode(e, 1, x.size)
    x.alignment != zero(Int32) && PB.encode(e, 2, x.alignment)
    !isempty(x.semantics) && PB.encode(e, 3, x.semantics)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"Node.XLAInstruction.LayoutAnalysis.Dimension")
    encoded_size = 0
    x.size != zero(Int32) && (encoded_size += PB._encoded_size(x.size, 1))
    x.alignment != zero(Int32) && (encoded_size += PB._encoded_size(x.alignment, 2))
    !isempty(x.semantics) && (encoded_size += PB._encoded_size(x.semantics, 3))
    return encoded_size
end

struct Metrics
    flops::Float64
    uncapped_flops::Float64
    bandwidth_utils::Vector{Float64}
    raw_time::Float64
    raw_flops::Float64
    bf16_flops::Float64
    normalized_time_ps::Float64
    raw_bytes_accessed_array::Vector{Float64}
    occurrences::UInt32
    avg_time_ps::Float64
end
Metrics(;flops = zero(Float64), uncapped_flops = zero(Float64), bandwidth_utils = Vector{Float64}(), raw_time = zero(Float64), raw_flops = zero(Float64), bf16_flops = zero(Float64), normalized_time_ps = zero(Float64), raw_bytes_accessed_array = Vector{Float64}(), occurrences = zero(UInt32), avg_time_ps = zero(Float64)) = Metrics(flops, uncapped_flops, bandwidth_utils, raw_time, raw_flops, bf16_flops, normalized_time_ps, raw_bytes_accessed_array, occurrences, avg_time_ps)
PB.reserved_fields(::Type{Metrics}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[1, 3, 4, 13, 14])
PB.default_values(::Type{Metrics}) = (;flops = zero(Float64), uncapped_flops = zero(Float64), bandwidth_utils = Vector{Float64}(), raw_time = zero(Float64), raw_flops = zero(Float64), bf16_flops = zero(Float64), normalized_time_ps = zero(Float64), raw_bytes_accessed_array = Vector{Float64}(), occurrences = zero(UInt32), avg_time_ps = zero(Float64))
PB.field_numbers(::Type{Metrics}) = (;flops = 2, uncapped_flops = 19, bandwidth_utils = 5, raw_time = 11, raw_flops = 12, bf16_flops = 18, normalized_time_ps = 20, raw_bytes_accessed_array = 15, occurrences = 16, avg_time_ps = 17)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Metrics})
    flops = zero(Float64)
    uncapped_flops = zero(Float64)
    bandwidth_utils = PB.BufferedVector{Float64}()
    raw_time = zero(Float64)
    raw_flops = zero(Float64)
    bf16_flops = zero(Float64)
    normalized_time_ps = zero(Float64)
    raw_bytes_accessed_array = PB.BufferedVector{Float64}()
    occurrences = zero(UInt32)
    avg_time_ps = zero(Float64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            flops = PB.decode(d, Float64)
        elseif field_number == 19
            uncapped_flops = PB.decode(d, Float64)
        elseif field_number == 5
            PB.decode!(d, wire_type, bandwidth_utils)
        elseif field_number == 11
            raw_time = PB.decode(d, Float64)
        elseif field_number == 12
            raw_flops = PB.decode(d, Float64)
        elseif field_number == 18
            bf16_flops = PB.decode(d, Float64)
        elseif field_number == 20
            normalized_time_ps = PB.decode(d, Float64)
        elseif field_number == 15
            PB.decode!(d, wire_type, raw_bytes_accessed_array)
        elseif field_number == 16
            occurrences = PB.decode(d, UInt32)
        elseif field_number == 17
            avg_time_ps = PB.decode(d, Float64)
        else
            Base.skip(d, wire_type)
        end
    end
    return Metrics(flops, uncapped_flops, bandwidth_utils[], raw_time, raw_flops, bf16_flops, normalized_time_ps, raw_bytes_accessed_array[], occurrences, avg_time_ps)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Metrics)
    initpos = position(e.io)
    x.flops !== zero(Float64) && PB.encode(e, 2, x.flops)
    x.uncapped_flops !== zero(Float64) && PB.encode(e, 19, x.uncapped_flops)
    !isempty(x.bandwidth_utils) && PB.encode(e, 5, x.bandwidth_utils)
    x.raw_time !== zero(Float64) && PB.encode(e, 11, x.raw_time)
    x.raw_flops !== zero(Float64) && PB.encode(e, 12, x.raw_flops)
    x.bf16_flops !== zero(Float64) && PB.encode(e, 18, x.bf16_flops)
    x.normalized_time_ps !== zero(Float64) && PB.encode(e, 20, x.normalized_time_ps)
    !isempty(x.raw_bytes_accessed_array) && PB.encode(e, 15, x.raw_bytes_accessed_array)
    x.occurrences != zero(UInt32) && PB.encode(e, 16, x.occurrences)
    x.avg_time_ps !== zero(Float64) && PB.encode(e, 17, x.avg_time_ps)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Metrics)
    encoded_size = 0
    x.flops !== zero(Float64) && (encoded_size += PB._encoded_size(x.flops, 2))
    x.uncapped_flops !== zero(Float64) && (encoded_size += PB._encoded_size(x.uncapped_flops, 19))
    !isempty(x.bandwidth_utils) && (encoded_size += PB._encoded_size(x.bandwidth_utils, 5))
    x.raw_time !== zero(Float64) && (encoded_size += PB._encoded_size(x.raw_time, 11))
    x.raw_flops !== zero(Float64) && (encoded_size += PB._encoded_size(x.raw_flops, 12))
    x.bf16_flops !== zero(Float64) && (encoded_size += PB._encoded_size(x.bf16_flops, 18))
    x.normalized_time_ps !== zero(Float64) && (encoded_size += PB._encoded_size(x.normalized_time_ps, 20))
    !isempty(x.raw_bytes_accessed_array) && (encoded_size += PB._encoded_size(x.raw_bytes_accessed_array, 15))
    x.occurrences != zero(UInt32) && (encoded_size += PB._encoded_size(x.occurrences, 16))
    x.avg_time_ps !== zero(Float64) && (encoded_size += PB._encoded_size(x.avg_time_ps, 17))
    return encoded_size
end

struct var"Node.XLAInstruction.LayoutAnalysis"
    dimensions::Vector{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}
end
var"Node.XLAInstruction.LayoutAnalysis"(;dimensions = Vector{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}()) = var"Node.XLAInstruction.LayoutAnalysis"(dimensions)
PB.default_values(::Type{var"Node.XLAInstruction.LayoutAnalysis"}) = (;dimensions = Vector{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}())
PB.field_numbers(::Type{var"Node.XLAInstruction.LayoutAnalysis"}) = (;dimensions = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"Node.XLAInstruction.LayoutAnalysis"})
    dimensions = PB.BufferedVector{var"Node.XLAInstruction.LayoutAnalysis.Dimension"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, dimensions)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"Node.XLAInstruction.LayoutAnalysis"(dimensions[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"Node.XLAInstruction.LayoutAnalysis")
    initpos = position(e.io)
    !isempty(x.dimensions) && PB.encode(e, 1, x.dimensions)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"Node.XLAInstruction.LayoutAnalysis")
    encoded_size = 0
    !isempty(x.dimensions) && (encoded_size += PB._encoded_size(x.dimensions, 1))
    return encoded_size
end

struct var"Node.XLAInstruction"
    op::String
    expression::String
    provenance::String
    category::String
    layout::Union{Nothing,var"Node.XLAInstruction.LayoutAnalysis"}
    computation_primitive_size::UInt32
    fingerprint::UInt64
    program_id::UInt64
    source_info::Union{Nothing,tensorflow.profiler.SourceInfo}
    xprof_kernel_metadata::String
end
var"Node.XLAInstruction"(;op = "", expression = "", provenance = "", category = "", layout = nothing, computation_primitive_size = zero(UInt32), fingerprint = zero(UInt64), program_id = zero(UInt64), source_info = nothing, xprof_kernel_metadata = "") = var"Node.XLAInstruction"(op, expression, provenance, category, layout, computation_primitive_size, fingerprint, program_id, source_info, xprof_kernel_metadata)
PB.default_values(::Type{var"Node.XLAInstruction"}) = (;op = "", expression = "", provenance = "", category = "", layout = nothing, computation_primitive_size = zero(UInt32), fingerprint = zero(UInt64), program_id = zero(UInt64), source_info = nothing, xprof_kernel_metadata = "")
PB.field_numbers(::Type{var"Node.XLAInstruction"}) = (;op = 1, expression = 2, provenance = 3, category = 4, layout = 5, computation_primitive_size = 6, fingerprint = 7, program_id = 8, source_info = 9, xprof_kernel_metadata = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"Node.XLAInstruction"})
    op = ""
    expression = ""
    provenance = ""
    category = ""
    layout = Ref{Union{Nothing,var"Node.XLAInstruction.LayoutAnalysis"}}(nothing)
    computation_primitive_size = zero(UInt32)
    fingerprint = zero(UInt64)
    program_id = zero(UInt64)
    source_info = Ref{Union{Nothing,tensorflow.profiler.SourceInfo}}(nothing)
    xprof_kernel_metadata = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            op = PB.decode(d, String)
        elseif field_number == 2
            expression = PB.decode(d, String)
        elseif field_number == 3
            provenance = PB.decode(d, String)
        elseif field_number == 4
            category = PB.decode(d, String)
        elseif field_number == 5
            PB.decode!(d, layout)
        elseif field_number == 6
            computation_primitive_size = PB.decode(d, UInt32)
        elseif field_number == 7
            fingerprint = PB.decode(d, UInt64)
        elseif field_number == 8
            program_id = PB.decode(d, UInt64)
        elseif field_number == 9
            PB.decode!(d, source_info)
        elseif field_number == 10
            xprof_kernel_metadata = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"Node.XLAInstruction"(op, expression, provenance, category, layout[], computation_primitive_size, fingerprint, program_id, source_info[], xprof_kernel_metadata)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"Node.XLAInstruction")
    initpos = position(e.io)
    !isempty(x.op) && PB.encode(e, 1, x.op)
    !isempty(x.expression) && PB.encode(e, 2, x.expression)
    !isempty(x.provenance) && PB.encode(e, 3, x.provenance)
    !isempty(x.category) && PB.encode(e, 4, x.category)
    !isnothing(x.layout) && PB.encode(e, 5, x.layout)
    x.computation_primitive_size != zero(UInt32) && PB.encode(e, 6, x.computation_primitive_size)
    x.fingerprint != zero(UInt64) && PB.encode(e, 7, x.fingerprint)
    x.program_id != zero(UInt64) && PB.encode(e, 8, x.program_id)
    !isnothing(x.source_info) && PB.encode(e, 9, x.source_info)
    !isempty(x.xprof_kernel_metadata) && PB.encode(e, 10, x.xprof_kernel_metadata)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"Node.XLAInstruction")
    encoded_size = 0
    !isempty(x.op) && (encoded_size += PB._encoded_size(x.op, 1))
    !isempty(x.expression) && (encoded_size += PB._encoded_size(x.expression, 2))
    !isempty(x.provenance) && (encoded_size += PB._encoded_size(x.provenance, 3))
    !isempty(x.category) && (encoded_size += PB._encoded_size(x.category, 4))
    !isnothing(x.layout) && (encoded_size += PB._encoded_size(x.layout, 5))
    x.computation_primitive_size != zero(UInt32) && (encoded_size += PB._encoded_size(x.computation_primitive_size, 6))
    x.fingerprint != zero(UInt64) && (encoded_size += PB._encoded_size(x.fingerprint, 7))
    x.program_id != zero(UInt64) && (encoded_size += PB._encoded_size(x.program_id, 8))
    !isnothing(x.source_info) && (encoded_size += PB._encoded_size(x.source_info, 9))
    !isempty(x.xprof_kernel_metadata) && (encoded_size += PB._encoded_size(x.xprof_kernel_metadata, 10))
    return encoded_size
end

struct Node
    name::String
    metrics::Union{Nothing,Metrics}
    children::Vector{Node}
    contents::Union{Nothing,OneOf{<:Union{var"Node.InstructionCategory",var"Node.XLAInstruction"}}}
    num_children::Int32
end
Node(;name = "", metrics = nothing, children = Vector{Node}(), contents = nothing, num_children = zero(Int32)) = Node(name, metrics, children, contents, num_children)
PB.oneof_field_types(::Type{Node}) = (;
    contents = (;category=var"Node.InstructionCategory", xla=var"Node.XLAInstruction"),
)
PB.default_values(::Type{Node}) = (;name = "", metrics = nothing, children = Vector{Node}(), category = nothing, xla = nothing, num_children = zero(Int32))
PB.field_numbers(::Type{Node}) = (;name = 1, metrics = 2, children = 3, category = 4, xla = 5, num_children = 6)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Node})
    name = ""
    metrics = Ref{Union{Nothing,Metrics}}(nothing)
    children = PB.BufferedVector{Node}()
    contents = nothing
    num_children = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, metrics)
        elseif field_number == 3
            PB.decode!(d, children)
        elseif field_number == 4
            contents = OneOf(:category, PB.decode(d, Ref{var"Node.InstructionCategory"}))
        elseif field_number == 5
            contents = OneOf(:xla, PB.decode(d, Ref{var"Node.XLAInstruction"}))
        elseif field_number == 6
            num_children = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return Node(name, metrics[], children[], contents, num_children)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Node)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isnothing(x.metrics) && PB.encode(e, 2, x.metrics)
    !isempty(x.children) && PB.encode(e, 3, x.children)
    if isnothing(x.contents);
    elseif x.contents.name === :category
        PB.encode(e, 4, x.contents[]::var"Node.InstructionCategory")
    elseif x.contents.name === :xla
        PB.encode(e, 5, x.contents[]::var"Node.XLAInstruction")
    end
    x.num_children != zero(Int32) && PB.encode(e, 6, x.num_children)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Node)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isnothing(x.metrics) && (encoded_size += PB._encoded_size(x.metrics, 2))
    !isempty(x.children) && (encoded_size += PB._encoded_size(x.children, 3))
    if isnothing(x.contents);
    elseif x.contents.name === :category
        encoded_size += PB._encoded_size(x.contents[]::var"Node.InstructionCategory", 4)
    elseif x.contents.name === :xla
        encoded_size += PB._encoded_size(x.contents[]::var"Node.XLAInstruction", 5)
    end
    x.num_children != zero(Int32) && (encoded_size += PB._encoded_size(x.num_children, 6))
    return encoded_size
end

struct Profile
    by_category::Union{Nothing,Node}
    by_program::Union{Nothing,Node}
    device_type::String
    agg_dvfs_time_scale_multiplier::Float64
    by_category_exclude_idle::Union{Nothing,Node}
    by_program_exclude_idle::Union{Nothing,Node}
    by_provenance::Union{Nothing,Node}
    by_provenance_exclude_idle::Union{Nothing,Node}
end
Profile(;by_category = nothing, by_program = nothing, device_type = "", agg_dvfs_time_scale_multiplier = zero(Float64), by_category_exclude_idle = nothing, by_program_exclude_idle = nothing, by_provenance = nothing, by_provenance_exclude_idle = nothing) = Profile(by_category, by_program, device_type, agg_dvfs_time_scale_multiplier, by_category_exclude_idle, by_program_exclude_idle, by_provenance, by_provenance_exclude_idle)
PB.reserved_fields(::Type{Profile}) = (names = ["by_program_structure", "per_program"], numbers = Union{Int,UnitRange{Int}}[2, 3])
PB.default_values(::Type{Profile}) = (;by_category = nothing, by_program = nothing, device_type = "", agg_dvfs_time_scale_multiplier = zero(Float64), by_category_exclude_idle = nothing, by_program_exclude_idle = nothing, by_provenance = nothing, by_provenance_exclude_idle = nothing)
PB.field_numbers(::Type{Profile}) = (;by_category = 1, by_program = 4, device_type = 5, agg_dvfs_time_scale_multiplier = 10, by_category_exclude_idle = 6, by_program_exclude_idle = 7, by_provenance = 8, by_provenance_exclude_idle = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Profile})
    by_category = Ref{Union{Nothing,Node}}(nothing)
    by_program = Ref{Union{Nothing,Node}}(nothing)
    device_type = ""
    agg_dvfs_time_scale_multiplier = zero(Float64)
    by_category_exclude_idle = Ref{Union{Nothing,Node}}(nothing)
    by_program_exclude_idle = Ref{Union{Nothing,Node}}(nothing)
    by_provenance = Ref{Union{Nothing,Node}}(nothing)
    by_provenance_exclude_idle = Ref{Union{Nothing,Node}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, by_category)
        elseif field_number == 4
            PB.decode!(d, by_program)
        elseif field_number == 5
            device_type = PB.decode(d, String)
        elseif field_number == 10
            agg_dvfs_time_scale_multiplier = PB.decode(d, Float64)
        elseif field_number == 6
            PB.decode!(d, by_category_exclude_idle)
        elseif field_number == 7
            PB.decode!(d, by_program_exclude_idle)
        elseif field_number == 8
            PB.decode!(d, by_provenance)
        elseif field_number == 9
            PB.decode!(d, by_provenance_exclude_idle)
        else
            Base.skip(d, wire_type)
        end
    end
    return Profile(by_category[], by_program[], device_type, agg_dvfs_time_scale_multiplier, by_category_exclude_idle[], by_program_exclude_idle[], by_provenance[], by_provenance_exclude_idle[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Profile)
    initpos = position(e.io)
    !isnothing(x.by_category) && PB.encode(e, 1, x.by_category)
    !isnothing(x.by_program) && PB.encode(e, 4, x.by_program)
    !isempty(x.device_type) && PB.encode(e, 5, x.device_type)
    x.agg_dvfs_time_scale_multiplier !== zero(Float64) && PB.encode(e, 10, x.agg_dvfs_time_scale_multiplier)
    !isnothing(x.by_category_exclude_idle) && PB.encode(e, 6, x.by_category_exclude_idle)
    !isnothing(x.by_program_exclude_idle) && PB.encode(e, 7, x.by_program_exclude_idle)
    !isnothing(x.by_provenance) && PB.encode(e, 8, x.by_provenance)
    !isnothing(x.by_provenance_exclude_idle) && PB.encode(e, 9, x.by_provenance_exclude_idle)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Profile)
    encoded_size = 0
    !isnothing(x.by_category) && (encoded_size += PB._encoded_size(x.by_category, 1))
    !isnothing(x.by_program) && (encoded_size += PB._encoded_size(x.by_program, 4))
    !isempty(x.device_type) && (encoded_size += PB._encoded_size(x.device_type, 5))
    x.agg_dvfs_time_scale_multiplier !== zero(Float64) && (encoded_size += PB._encoded_size(x.agg_dvfs_time_scale_multiplier, 10))
    !isnothing(x.by_category_exclude_idle) && (encoded_size += PB._encoded_size(x.by_category_exclude_idle, 6))
    !isnothing(x.by_program_exclude_idle) && (encoded_size += PB._encoded_size(x.by_program_exclude_idle, 7))
    !isnothing(x.by_provenance) && (encoded_size += PB._encoded_size(x.by_provenance, 8))
    !isnothing(x.by_provenance_exclude_idle) && (encoded_size += PB._encoded_size(x.by_provenance_exclude_idle, 9))
    return encoded_size
end
