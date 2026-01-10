import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export OptionOverrideProto, ExecutableBuildOptionsProto, CompileOptionsProto
export ExecutableAndOptionsProto


struct OptionOverrideProto
    value::Union{Nothing,OneOf{<:Union{String,Bool,Int64,Float64}}}
end
OptionOverrideProto(;value = nothing) = OptionOverrideProto(value)
PB.oneof_field_types(::Type{OptionOverrideProto}) = (;
    value = (;string_field=String, bool_field=Bool, int_field=Int64, double_field=Float64),
)
PB.default_values(::Type{OptionOverrideProto}) = (;string_field = "", bool_field = false, int_field = zero(Int64), double_field = zero(Float64))
PB.field_numbers(::Type{OptionOverrideProto}) = (;string_field = 1, bool_field = 2, int_field = 3, double_field = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:OptionOverrideProto})
    value = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            value = OneOf(:string_field, PB.decode(d, String))
        elseif field_number == 2
            value = OneOf(:bool_field, PB.decode(d, Bool))
        elseif field_number == 3
            value = OneOf(:int_field, PB.decode(d, Int64))
        elseif field_number == 4
            value = OneOf(:double_field, PB.decode(d, Float64))
        else
            Base.skip(d, wire_type)
        end
    end
    return OptionOverrideProto(value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::OptionOverrideProto)
    initpos = position(e.io)
    if isnothing(x.value);
    elseif x.value.name === :string_field
        PB.encode(e, 1, x.value[]::String)
    elseif x.value.name === :bool_field
        PB.encode(e, 2, x.value[]::Bool)
    elseif x.value.name === :int_field
        PB.encode(e, 3, x.value[]::Int64)
    elseif x.value.name === :double_field
        PB.encode(e, 4, x.value[]::Float64)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::OptionOverrideProto)
    encoded_size = 0
    if isnothing(x.value);
    elseif x.value.name === :string_field
        encoded_size += PB._encoded_size(x.value[]::String, 1)
    elseif x.value.name === :bool_field
        encoded_size += PB._encoded_size(x.value[]::Bool, 2)
    elseif x.value.name === :int_field
        encoded_size += PB._encoded_size(x.value[]::Int64, 3)
    elseif x.value.name === :double_field
        encoded_size += PB._encoded_size(x.value[]::Float64, 4)
    end
    return encoded_size
end

struct ExecutableBuildOptionsProto
    device_ordinal::Int64
    result_layout::Union{Nothing,ShapeProto}
    comp_envs::Union{Nothing,CompilationEnvironmentsProto}
    debug_options::Union{Nothing,DebugOptions}
    num_replicas::Int64
    num_partitions::Int64
    use_spmd_partitioning::Bool
    use_auto_spmd_partitioning::Bool
    exec_time_optimization_effort::Float32
    memory_fitting_effort::Float32
    optimization_level::var"ExecutionOptions.EffortLevel".T
    memory_fitting_level::var"ExecutionOptions.EffortLevel".T
    deduplicate_hlo::Bool
    device_assignment::Union{Nothing,DeviceAssignmentProto}
    alias_passthrough_params::Bool
    run_backend_only::Bool
    allow_spmd_sharding_propagation_to_parameters::Vector{Bool}
    allow_spmd_sharding_propagation_to_output::Vector{Bool}
    fdo_profile::Vector{UInt8}
    device_memory_size::Int64
    auto_spmd_partitioning_mesh_shape::Vector{Int64}
    auto_spmd_partitioning_mesh_ids::Vector{Int64}
    use_shardy_partitioner::Bool
    process_index::Int64
    process_count::Int64
    slice_size::Int64
end
ExecutableBuildOptionsProto(;device_ordinal = zero(Int64), result_layout = nothing, comp_envs = nothing, debug_options = nothing, num_replicas = zero(Int64), num_partitions = zero(Int64), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, device_assignment = nothing, alias_passthrough_params = false, run_backend_only = false, allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), fdo_profile = UInt8[], device_memory_size = zero(Int64), auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), use_shardy_partitioner = false, process_index = zero(Int64), process_count = zero(Int64), slice_size = zero(Int64)) = ExecutableBuildOptionsProto(device_ordinal, result_layout, comp_envs, debug_options, num_replicas, num_partitions, use_spmd_partitioning, use_auto_spmd_partitioning, exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, device_assignment, alias_passthrough_params, run_backend_only, allow_spmd_sharding_propagation_to_parameters, allow_spmd_sharding_propagation_to_output, fdo_profile, device_memory_size, auto_spmd_partitioning_mesh_shape, auto_spmd_partitioning_mesh_ids, use_shardy_partitioner, process_index, process_count, slice_size)
PB.default_values(::Type{ExecutableBuildOptionsProto}) = (;device_ordinal = zero(Int64), result_layout = nothing, comp_envs = nothing, debug_options = nothing, num_replicas = zero(Int64), num_partitions = zero(Int64), use_spmd_partitioning = false, use_auto_spmd_partitioning = false, exec_time_optimization_effort = zero(Float32), memory_fitting_effort = zero(Float32), optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN, deduplicate_hlo = false, device_assignment = nothing, alias_passthrough_params = false, run_backend_only = false, allow_spmd_sharding_propagation_to_parameters = Vector{Bool}(), allow_spmd_sharding_propagation_to_output = Vector{Bool}(), fdo_profile = UInt8[], device_memory_size = zero(Int64), auto_spmd_partitioning_mesh_shape = Vector{Int64}(), auto_spmd_partitioning_mesh_ids = Vector{Int64}(), use_shardy_partitioner = false, process_index = zero(Int64), process_count = zero(Int64), slice_size = zero(Int64))
PB.field_numbers(::Type{ExecutableBuildOptionsProto}) = (;device_ordinal = 1, result_layout = 2, comp_envs = 13, debug_options = 3, num_replicas = 4, num_partitions = 5, use_spmd_partitioning = 6, use_auto_spmd_partitioning = 7, exec_time_optimization_effort = 20, memory_fitting_effort = 21, optimization_level = 24, memory_fitting_level = 25, deduplicate_hlo = 8, device_assignment = 9, alias_passthrough_params = 10, run_backend_only = 11, allow_spmd_sharding_propagation_to_parameters = 18, allow_spmd_sharding_propagation_to_output = 12, fdo_profile = 14, device_memory_size = 15, auto_spmd_partitioning_mesh_shape = 16, auto_spmd_partitioning_mesh_ids = 17, use_shardy_partitioner = 19, process_index = 22, process_count = 23, slice_size = 26)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecutableBuildOptionsProto})
    device_ordinal = zero(Int64)
    result_layout = Ref{Union{Nothing,ShapeProto}}(nothing)
    comp_envs = Ref{Union{Nothing,CompilationEnvironmentsProto}}(nothing)
    debug_options = Ref{Union{Nothing,DebugOptions}}(nothing)
    num_replicas = zero(Int64)
    num_partitions = zero(Int64)
    use_spmd_partitioning = false
    use_auto_spmd_partitioning = false
    exec_time_optimization_effort = zero(Float32)
    memory_fitting_effort = zero(Float32)
    optimization_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    memory_fitting_level = var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN
    deduplicate_hlo = false
    device_assignment = Ref{Union{Nothing,DeviceAssignmentProto}}(nothing)
    alias_passthrough_params = false
    run_backend_only = false
    allow_spmd_sharding_propagation_to_parameters = PB.BufferedVector{Bool}()
    allow_spmd_sharding_propagation_to_output = PB.BufferedVector{Bool}()
    fdo_profile = UInt8[]
    device_memory_size = zero(Int64)
    auto_spmd_partitioning_mesh_shape = PB.BufferedVector{Int64}()
    auto_spmd_partitioning_mesh_ids = PB.BufferedVector{Int64}()
    use_shardy_partitioner = false
    process_index = zero(Int64)
    process_count = zero(Int64)
    slice_size = zero(Int64)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            device_ordinal = PB.decode(d, Int64)
        elseif field_number == 2
            PB.decode!(d, result_layout)
        elseif field_number == 13
            PB.decode!(d, comp_envs)
        elseif field_number == 3
            PB.decode!(d, debug_options)
        elseif field_number == 4
            num_replicas = PB.decode(d, Int64)
        elseif field_number == 5
            num_partitions = PB.decode(d, Int64)
        elseif field_number == 6
            use_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 7
            use_auto_spmd_partitioning = PB.decode(d, Bool)
        elseif field_number == 20
            exec_time_optimization_effort = PB.decode(d, Float32)
        elseif field_number == 21
            memory_fitting_effort = PB.decode(d, Float32)
        elseif field_number == 24
            optimization_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 25
            memory_fitting_level = PB.decode(d, var"ExecutionOptions.EffortLevel".T)
        elseif field_number == 8
            deduplicate_hlo = PB.decode(d, Bool)
        elseif field_number == 9
            PB.decode!(d, device_assignment)
        elseif field_number == 10
            alias_passthrough_params = PB.decode(d, Bool)
        elseif field_number == 11
            run_backend_only = PB.decode(d, Bool)
        elseif field_number == 18
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_parameters)
        elseif field_number == 12
            PB.decode!(d, wire_type, allow_spmd_sharding_propagation_to_output)
        elseif field_number == 14
            fdo_profile = PB.decode(d, Vector{UInt8})
        elseif field_number == 15
            device_memory_size = PB.decode(d, Int64)
        elseif field_number == 16
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_shape)
        elseif field_number == 17
            PB.decode!(d, wire_type, auto_spmd_partitioning_mesh_ids)
        elseif field_number == 19
            use_shardy_partitioner = PB.decode(d, Bool)
        elseif field_number == 22
            process_index = PB.decode(d, Int64)
        elseif field_number == 23
            process_count = PB.decode(d, Int64)
        elseif field_number == 26
            slice_size = PB.decode(d, Int64)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecutableBuildOptionsProto(device_ordinal, result_layout[], comp_envs[], debug_options[], num_replicas, num_partitions, use_spmd_partitioning, use_auto_spmd_partitioning, exec_time_optimization_effort, memory_fitting_effort, optimization_level, memory_fitting_level, deduplicate_hlo, device_assignment[], alias_passthrough_params, run_backend_only, allow_spmd_sharding_propagation_to_parameters[], allow_spmd_sharding_propagation_to_output[], fdo_profile, device_memory_size, auto_spmd_partitioning_mesh_shape[], auto_spmd_partitioning_mesh_ids[], use_shardy_partitioner, process_index, process_count, slice_size)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecutableBuildOptionsProto)
    initpos = position(e.io)
    x.device_ordinal != zero(Int64) && PB.encode(e, 1, x.device_ordinal)
    !isnothing(x.result_layout) && PB.encode(e, 2, x.result_layout)
    !isnothing(x.comp_envs) && PB.encode(e, 13, x.comp_envs)
    !isnothing(x.debug_options) && PB.encode(e, 3, x.debug_options)
    x.num_replicas != zero(Int64) && PB.encode(e, 4, x.num_replicas)
    x.num_partitions != zero(Int64) && PB.encode(e, 5, x.num_partitions)
    x.use_spmd_partitioning != false && PB.encode(e, 6, x.use_spmd_partitioning)
    x.use_auto_spmd_partitioning != false && PB.encode(e, 7, x.use_auto_spmd_partitioning)
    x.exec_time_optimization_effort !== zero(Float32) && PB.encode(e, 20, x.exec_time_optimization_effort)
    x.memory_fitting_effort !== zero(Float32) && PB.encode(e, 21, x.memory_fitting_effort)
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 24, x.optimization_level)
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && PB.encode(e, 25, x.memory_fitting_level)
    x.deduplicate_hlo != false && PB.encode(e, 8, x.deduplicate_hlo)
    !isnothing(x.device_assignment) && PB.encode(e, 9, x.device_assignment)
    x.alias_passthrough_params != false && PB.encode(e, 10, x.alias_passthrough_params)
    x.run_backend_only != false && PB.encode(e, 11, x.run_backend_only)
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && PB.encode(e, 18, x.allow_spmd_sharding_propagation_to_parameters)
    !isempty(x.allow_spmd_sharding_propagation_to_output) && PB.encode(e, 12, x.allow_spmd_sharding_propagation_to_output)
    !isempty(x.fdo_profile) && PB.encode(e, 14, x.fdo_profile)
    x.device_memory_size != zero(Int64) && PB.encode(e, 15, x.device_memory_size)
    !isempty(x.auto_spmd_partitioning_mesh_shape) && PB.encode(e, 16, x.auto_spmd_partitioning_mesh_shape)
    !isempty(x.auto_spmd_partitioning_mesh_ids) && PB.encode(e, 17, x.auto_spmd_partitioning_mesh_ids)
    x.use_shardy_partitioner != false && PB.encode(e, 19, x.use_shardy_partitioner)
    x.process_index != zero(Int64) && PB.encode(e, 22, x.process_index)
    x.process_count != zero(Int64) && PB.encode(e, 23, x.process_count)
    x.slice_size != zero(Int64) && PB.encode(e, 26, x.slice_size)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecutableBuildOptionsProto)
    encoded_size = 0
    x.device_ordinal != zero(Int64) && (encoded_size += PB._encoded_size(x.device_ordinal, 1))
    !isnothing(x.result_layout) && (encoded_size += PB._encoded_size(x.result_layout, 2))
    !isnothing(x.comp_envs) && (encoded_size += PB._encoded_size(x.comp_envs, 13))
    !isnothing(x.debug_options) && (encoded_size += PB._encoded_size(x.debug_options, 3))
    x.num_replicas != zero(Int64) && (encoded_size += PB._encoded_size(x.num_replicas, 4))
    x.num_partitions != zero(Int64) && (encoded_size += PB._encoded_size(x.num_partitions, 5))
    x.use_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_spmd_partitioning, 6))
    x.use_auto_spmd_partitioning != false && (encoded_size += PB._encoded_size(x.use_auto_spmd_partitioning, 7))
    x.exec_time_optimization_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.exec_time_optimization_effort, 20))
    x.memory_fitting_effort !== zero(Float32) && (encoded_size += PB._encoded_size(x.memory_fitting_effort, 21))
    x.optimization_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.optimization_level, 24))
    x.memory_fitting_level != var"ExecutionOptions.EffortLevel".EFFORT_UNKNOWN && (encoded_size += PB._encoded_size(x.memory_fitting_level, 25))
    x.deduplicate_hlo != false && (encoded_size += PB._encoded_size(x.deduplicate_hlo, 8))
    !isnothing(x.device_assignment) && (encoded_size += PB._encoded_size(x.device_assignment, 9))
    x.alias_passthrough_params != false && (encoded_size += PB._encoded_size(x.alias_passthrough_params, 10))
    x.run_backend_only != false && (encoded_size += PB._encoded_size(x.run_backend_only, 11))
    !isempty(x.allow_spmd_sharding_propagation_to_parameters) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_parameters, 18))
    !isempty(x.allow_spmd_sharding_propagation_to_output) && (encoded_size += PB._encoded_size(x.allow_spmd_sharding_propagation_to_output, 12))
    !isempty(x.fdo_profile) && (encoded_size += PB._encoded_size(x.fdo_profile, 14))
    x.device_memory_size != zero(Int64) && (encoded_size += PB._encoded_size(x.device_memory_size, 15))
    !isempty(x.auto_spmd_partitioning_mesh_shape) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_shape, 16))
    !isempty(x.auto_spmd_partitioning_mesh_ids) && (encoded_size += PB._encoded_size(x.auto_spmd_partitioning_mesh_ids, 17))
    x.use_shardy_partitioner != false && (encoded_size += PB._encoded_size(x.use_shardy_partitioner, 19))
    x.process_index != zero(Int64) && (encoded_size += PB._encoded_size(x.process_index, 22))
    x.process_count != zero(Int64) && (encoded_size += PB._encoded_size(x.process_count, 23))
    x.slice_size != zero(Int64) && (encoded_size += PB._encoded_size(x.slice_size, 26))
    return encoded_size
end

struct CompileOptionsProto
    argument_layouts::Vector{ShapeProto}
    parameter_is_tupled_arguments::Bool
    executable_build_options::Union{Nothing,ExecutableBuildOptionsProto}
    compile_portable_executable::Bool
    profile_version::Int64
    serialized_multi_slice_config::Vector{UInt8}
    env_option_overrides::Dict{String,OptionOverrideProto}
    target_config::Union{Nothing,stream_executor.GpuTargetConfigProto}
    allow_in_place_mlir_modification::Bool
    matrix_unit_operand_precision::var"PrecisionConfig.Precision".T
end
CompileOptionsProto(;argument_layouts = Vector{ShapeProto}(), parameter_is_tupled_arguments = false, executable_build_options = nothing, compile_portable_executable = false, profile_version = zero(Int64), serialized_multi_slice_config = UInt8[], env_option_overrides = Dict{String,OptionOverrideProto}(), target_config = nothing, allow_in_place_mlir_modification = false, matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT) = CompileOptionsProto(argument_layouts, parameter_is_tupled_arguments, executable_build_options, compile_portable_executable, profile_version, serialized_multi_slice_config, env_option_overrides, target_config, allow_in_place_mlir_modification, matrix_unit_operand_precision)
PB.default_values(::Type{CompileOptionsProto}) = (;argument_layouts = Vector{ShapeProto}(), parameter_is_tupled_arguments = false, executable_build_options = nothing, compile_portable_executable = false, profile_version = zero(Int64), serialized_multi_slice_config = UInt8[], env_option_overrides = Dict{String,OptionOverrideProto}(), target_config = nothing, allow_in_place_mlir_modification = false, matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT)
PB.field_numbers(::Type{CompileOptionsProto}) = (;argument_layouts = 1, parameter_is_tupled_arguments = 2, executable_build_options = 3, compile_portable_executable = 4, profile_version = 5, serialized_multi_slice_config = 6, env_option_overrides = 7, target_config = 8, allow_in_place_mlir_modification = 9, matrix_unit_operand_precision = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CompileOptionsProto})
    argument_layouts = PB.BufferedVector{ShapeProto}()
    parameter_is_tupled_arguments = false
    executable_build_options = Ref{Union{Nothing,ExecutableBuildOptionsProto}}(nothing)
    compile_portable_executable = false
    profile_version = zero(Int64)
    serialized_multi_slice_config = UInt8[]
    env_option_overrides = Dict{String,OptionOverrideProto}()
    target_config = Ref{Union{Nothing,stream_executor.GpuTargetConfigProto}}(nothing)
    allow_in_place_mlir_modification = false
    matrix_unit_operand_precision = var"PrecisionConfig.Precision".DEFAULT
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, argument_layouts)
        elseif field_number == 2
            parameter_is_tupled_arguments = PB.decode(d, Bool)
        elseif field_number == 3
            PB.decode!(d, executable_build_options)
        elseif field_number == 4
            compile_portable_executable = PB.decode(d, Bool)
        elseif field_number == 5
            profile_version = PB.decode(d, Int64)
        elseif field_number == 6
            serialized_multi_slice_config = PB.decode(d, Vector{UInt8})
        elseif field_number == 7
            PB.decode!(d, env_option_overrides)
        elseif field_number == 8
            PB.decode!(d, target_config)
        elseif field_number == 9
            allow_in_place_mlir_modification = PB.decode(d, Bool)
        elseif field_number == 10
            matrix_unit_operand_precision = PB.decode(d, var"PrecisionConfig.Precision".T)
        else
            Base.skip(d, wire_type)
        end
    end
    return CompileOptionsProto(argument_layouts[], parameter_is_tupled_arguments, executable_build_options[], compile_portable_executable, profile_version, serialized_multi_slice_config, env_option_overrides, target_config[], allow_in_place_mlir_modification, matrix_unit_operand_precision)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CompileOptionsProto)
    initpos = position(e.io)
    !isempty(x.argument_layouts) && PB.encode(e, 1, x.argument_layouts)
    x.parameter_is_tupled_arguments != false && PB.encode(e, 2, x.parameter_is_tupled_arguments)
    !isnothing(x.executable_build_options) && PB.encode(e, 3, x.executable_build_options)
    x.compile_portable_executable != false && PB.encode(e, 4, x.compile_portable_executable)
    x.profile_version != zero(Int64) && PB.encode(e, 5, x.profile_version)
    !isempty(x.serialized_multi_slice_config) && PB.encode(e, 6, x.serialized_multi_slice_config)
    !isempty(x.env_option_overrides) && PB.encode(e, 7, x.env_option_overrides)
    !isnothing(x.target_config) && PB.encode(e, 8, x.target_config)
    x.allow_in_place_mlir_modification != false && PB.encode(e, 9, x.allow_in_place_mlir_modification)
    x.matrix_unit_operand_precision != var"PrecisionConfig.Precision".DEFAULT && PB.encode(e, 10, x.matrix_unit_operand_precision)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CompileOptionsProto)
    encoded_size = 0
    !isempty(x.argument_layouts) && (encoded_size += PB._encoded_size(x.argument_layouts, 1))
    x.parameter_is_tupled_arguments != false && (encoded_size += PB._encoded_size(x.parameter_is_tupled_arguments, 2))
    !isnothing(x.executable_build_options) && (encoded_size += PB._encoded_size(x.executable_build_options, 3))
    x.compile_portable_executable != false && (encoded_size += PB._encoded_size(x.compile_portable_executable, 4))
    x.profile_version != zero(Int64) && (encoded_size += PB._encoded_size(x.profile_version, 5))
    !isempty(x.serialized_multi_slice_config) && (encoded_size += PB._encoded_size(x.serialized_multi_slice_config, 6))
    !isempty(x.env_option_overrides) && (encoded_size += PB._encoded_size(x.env_option_overrides, 7))
    !isnothing(x.target_config) && (encoded_size += PB._encoded_size(x.target_config, 8))
    x.allow_in_place_mlir_modification != false && (encoded_size += PB._encoded_size(x.allow_in_place_mlir_modification, 9))
    x.matrix_unit_operand_precision != var"PrecisionConfig.Precision".DEFAULT && (encoded_size += PB._encoded_size(x.matrix_unit_operand_precision, 10))
    return encoded_size
end

struct ExecutableAndOptionsProto
    serialized_executable::Vector{UInt8}
    compile_options::Union{Nothing,CompileOptionsProto}
    pjrt_client_name::String
end
ExecutableAndOptionsProto(;serialized_executable = UInt8[], compile_options = nothing, pjrt_client_name = "") = ExecutableAndOptionsProto(serialized_executable, compile_options, pjrt_client_name)
PB.default_values(::Type{ExecutableAndOptionsProto}) = (;serialized_executable = UInt8[], compile_options = nothing, pjrt_client_name = "")
PB.field_numbers(::Type{ExecutableAndOptionsProto}) = (;serialized_executable = 1, compile_options = 2, pjrt_client_name = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:ExecutableAndOptionsProto})
    serialized_executable = UInt8[]
    compile_options = Ref{Union{Nothing,CompileOptionsProto}}(nothing)
    pjrt_client_name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            serialized_executable = PB.decode(d, Vector{UInt8})
        elseif field_number == 2
            PB.decode!(d, compile_options)
        elseif field_number == 3
            pjrt_client_name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return ExecutableAndOptionsProto(serialized_executable, compile_options[], pjrt_client_name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::ExecutableAndOptionsProto)
    initpos = position(e.io)
    !isempty(x.serialized_executable) && PB.encode(e, 1, x.serialized_executable)
    !isnothing(x.compile_options) && PB.encode(e, 2, x.compile_options)
    !isempty(x.pjrt_client_name) && PB.encode(e, 3, x.pjrt_client_name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::ExecutableAndOptionsProto)
    encoded_size = 0
    !isempty(x.serialized_executable) && (encoded_size += PB._encoded_size(x.serialized_executable, 1))
    !isnothing(x.compile_options) && (encoded_size += PB._encoded_size(x.compile_options, 2))
    !isempty(x.pjrt_client_name) && (encoded_size += PB._encoded_size(x.pjrt_client_name, 3))
    return encoded_size
end
