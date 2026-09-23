import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"CompilationResultProto.ObjFileKind", TargetMachineOptionsProto
export var"SymbolProto.FunctionTypeId", ObjFileProto, SymbolProto, CompilationResultProto


@enumx var"CompilationResultProto.ObjFileKind" UNKNOWN=0 CLASSIC=1 KERNELS=2

struct TargetMachineOptionsProto
    triple::String
    cpu::String
    features::String
end
PB.default_values(::Type{TargetMachineOptionsProto}) = (;triple = "", cpu = "", features = "")
PB.field_numbers(::Type{TargetMachineOptionsProto}) = (;triple = 1, cpu = 2, features = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TargetMachineOptionsProto}, _endpos::Int=0, _group::Bool=false)
    triple = ""
    cpu = ""
    features = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            triple = PB.decode(d, String)
        elseif field_number == 2
            cpu = PB.decode(d, String)
        elseif field_number == 3
            features = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return TargetMachineOptionsProto(triple, cpu, features)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TargetMachineOptionsProto)
    initpos = position(e.io)
    !isempty(x.triple) && PB.encode(e, 1, x.triple)
    !isempty(x.cpu) && PB.encode(e, 2, x.cpu)
    !isempty(x.features) && PB.encode(e, 3, x.features)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TargetMachineOptionsProto)
    encoded_size = 0
    !isempty(x.triple) && (encoded_size += PB._encoded_size(x.triple, 1))
    !isempty(x.cpu) && (encoded_size += PB._encoded_size(x.cpu, 2))
    !isempty(x.features) && (encoded_size += PB._encoded_size(x.features, 3))
    return encoded_size
end

@enumx var"SymbolProto.FunctionTypeId" UNKNOWN=0 KERNEL=1 COMPARATOR=2

struct ObjFileProto
    contents::Vector{UInt8}
    name::String
end
PB.default_values(::Type{ObjFileProto}) = (;contents = UInt8[], name = "")
PB.field_numbers(::Type{ObjFileProto}) = (;contents = 1, name = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ObjFileProto}, _endpos::Int=0, _group::Bool=false)
    contents = UInt8[]
    name = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            contents = PB.decode(d, Vector{UInt8})
        elseif field_number == 2
            name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ObjFileProto(contents, name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ObjFileProto)
    initpos = position(e.io)
    !isempty(x.contents) && PB.encode(e, 1, x.contents)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ObjFileProto)
    encoded_size = 0
    !isempty(x.contents) && (encoded_size += PB._encoded_size(x.contents, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    return encoded_size
end

struct SymbolProto
    function_type_id::var"SymbolProto.FunctionTypeId".T
    name::String
end
PB.default_values(::Type{SymbolProto}) = (;function_type_id = var"SymbolProto.FunctionTypeId".UNKNOWN, name = "")
PB.field_numbers(::Type{SymbolProto}) = (;function_type_id = 1, name = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SymbolProto}, _endpos::Int=0, _group::Bool=false)
    function_type_id = var"SymbolProto.FunctionTypeId".UNKNOWN
    name = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            function_type_id = PB.decode(d, var"SymbolProto.FunctionTypeId".T)
        elseif field_number == 2
            name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return SymbolProto(function_type_id, name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SymbolProto)
    initpos = position(e.io)
    x.function_type_id != var"SymbolProto.FunctionTypeId".UNKNOWN && PB.encode(e, 1, x.function_type_id)
    !isempty(x.name) && PB.encode(e, 2, x.name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SymbolProto)
    encoded_size = 0
    x.function_type_id != var"SymbolProto.FunctionTypeId".UNKNOWN && (encoded_size += PB._encoded_size(x.function_type_id, 1))
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 2))
    return encoded_size
end

struct CompilationResultProto
    hlo_module::Union{Nothing,xla.HloModuleProtoWithConfig}
    buffer_assignment::Union{Nothing,xla.BufferAssignmentProto}
    entry_function_name::String
    obj_files_kind::var"CompilationResultProto.ObjFileKind".T
    thunk_sequence::Union{Nothing,ThunkSequenceProto}
    compiled_symbols::Vector{SymbolProto}
    object_files::Vector{ObjFileProto}
    target_machine_options::Union{Nothing,TargetMachineOptionsProto}
    data_layout::String
end
PB.reserved_fields(::Type{CompilationResultProto}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[4])
PB.default_values(::Type{CompilationResultProto}) = (;hlo_module = nothing, buffer_assignment = nothing, entry_function_name = "", obj_files_kind = var"CompilationResultProto.ObjFileKind".UNKNOWN, thunk_sequence = nothing, compiled_symbols = Vector{SymbolProto}(), object_files = Vector{ObjFileProto}(), target_machine_options = nothing, data_layout = "")
PB.field_numbers(::Type{CompilationResultProto}) = (;hlo_module = 1, buffer_assignment = 2, entry_function_name = 3, obj_files_kind = 5, thunk_sequence = 6, compiled_symbols = 7, object_files = 8, target_machine_options = 9, data_layout = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CompilationResultProto}, _endpos::Int=0, _group::Bool=false)
    hlo_module = Ref{Union{Nothing,xla.HloModuleProtoWithConfig}}(nothing)
    buffer_assignment = Ref{Union{Nothing,xla.BufferAssignmentProto}}(nothing)
    entry_function_name = ""
    obj_files_kind = var"CompilationResultProto.ObjFileKind".UNKNOWN
    thunk_sequence = Ref{Union{Nothing,ThunkSequenceProto}}(nothing)
    compiled_symbols = PB.BufferedVector{SymbolProto}()
    object_files = PB.BufferedVector{ObjFileProto}()
    target_machine_options = Ref{Union{Nothing,TargetMachineOptionsProto}}(nothing)
    data_layout = ""
    while !PB.message_done(d, _endpos, _group)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, hlo_module)
        elseif field_number == 2
            PB.decode!(d, buffer_assignment)
        elseif field_number == 3
            entry_function_name = PB.decode(d, String)
        elseif field_number == 5
            obj_files_kind = PB.decode(d, var"CompilationResultProto.ObjFileKind".T)
        elseif field_number == 6
            PB.decode!(d, thunk_sequence)
        elseif field_number == 7
            PB.decode!(d, compiled_symbols)
        elseif field_number == 8
            PB.decode!(d, object_files)
        elseif field_number == 9
            PB.decode!(d, target_machine_options)
        elseif field_number == 10
            data_layout = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return CompilationResultProto(hlo_module[], buffer_assignment[], entry_function_name, obj_files_kind, thunk_sequence[], compiled_symbols[], object_files[], target_machine_options[], data_layout)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CompilationResultProto)
    initpos = position(e.io)
    !isnothing(x.hlo_module) && PB.encode(e, 1, x.hlo_module)
    !isnothing(x.buffer_assignment) && PB.encode(e, 2, x.buffer_assignment)
    !isempty(x.entry_function_name) && PB.encode(e, 3, x.entry_function_name)
    x.obj_files_kind != var"CompilationResultProto.ObjFileKind".UNKNOWN && PB.encode(e, 5, x.obj_files_kind)
    !isnothing(x.thunk_sequence) && PB.encode(e, 6, x.thunk_sequence)
    !isempty(x.compiled_symbols) && PB.encode(e, 7, x.compiled_symbols)
    !isempty(x.object_files) && PB.encode(e, 8, x.object_files)
    !isnothing(x.target_machine_options) && PB.encode(e, 9, x.target_machine_options)
    !isempty(x.data_layout) && PB.encode(e, 10, x.data_layout)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CompilationResultProto)
    encoded_size = 0
    !isnothing(x.hlo_module) && (encoded_size += PB._encoded_size(x.hlo_module, 1))
    !isnothing(x.buffer_assignment) && (encoded_size += PB._encoded_size(x.buffer_assignment, 2))
    !isempty(x.entry_function_name) && (encoded_size += PB._encoded_size(x.entry_function_name, 3))
    x.obj_files_kind != var"CompilationResultProto.ObjFileKind".UNKNOWN && (encoded_size += PB._encoded_size(x.obj_files_kind, 5))
    !isnothing(x.thunk_sequence) && (encoded_size += PB._encoded_size(x.thunk_sequence, 6))
    !isempty(x.compiled_symbols) && (encoded_size += PB._encoded_size(x.compiled_symbols, 7))
    !isempty(x.object_files) && (encoded_size += PB._encoded_size(x.object_files, 8))
    !isnothing(x.target_machine_options) && (encoded_size += PB._encoded_size(x.target_machine_options, 9))
    !isempty(x.data_layout) && (encoded_size += PB._encoded_size(x.data_layout, 10))
    return encoded_size
end
