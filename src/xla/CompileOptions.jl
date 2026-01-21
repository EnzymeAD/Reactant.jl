function get_default_debug_options()
    size = Ref{Csize_t}(0)
    data = @ccall MLIR.API.mlir_c.ReactantGetDebugOptions(size::Ptr{Csize_t})::Ptr{UInt8}
    bytes = unsafe_wrap(Array, data, (size[],); own=false)
    proto = Reactant.ProtoUtils.proto_from_bytes(Reactant.Proto.xla.DebugOptions, bytes)
    @ccall free(data::Ptr{UInt8})::Cvoid
    return proto
end

function get_default_compile_options()
    size = Ref{Csize_t}(0)
    data = @ccall MLIR.API.mlir_c.ReactantGetCompileOptions(size::Ptr{Csize_t})::Ptr{UInt8}
    bytes = unsafe_wrap(Array, data, (size[],); own=false)
    proto = Reactant.ProtoUtils.proto_from_bytes(
        Reactant.Proto.xla.CompileOptionsProto, bytes
    )
    @ccall free(data::Ptr{UInt8})::Cvoid
    return proto
end

function get_debug_options(; kwargs...)
    debug_options = get_default_debug_options()

    # default overrides. can be changed by the user by passing in kwargs
    debug_options.xla_gpu_cuda_data_dir = CUDA_DATA_DIR[]
    debug_options.xla_enable_enzyme_comms_opt = true
    debug_options.xla_gpu_experimental_use_raft_select_k = true

    if Reactant.PersistentCompileCache.kernel_cache_enabled()
        debug_options.xla_gpu_kernel_cache_file = Reactant.PersistentCompileCache.get_kernel_cache_path()
        debug_options.xla_gpu_enable_llvm_module_compilation_parallelism = true
    end
    if Reactant.PersistentCompileCache.autotune_cache_enabled()
        debug_options.xla_gpu_per_fusion_autotune_cache_dir = Reactant.PersistentCompileCache.get_autotune_cache_directory()
        if Reactant.Distributed.local_rank() <= 0
            debug_options.xla_gpu_experimental_autotune_cache_mode =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UPDATE
        else
            debug_options.xla_gpu_experimental_autotune_cache_mode =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_READ
        end
    end

    for (key, value) in pairs(kwargs)
        setproperty!(debug_options, key, value)
    end

    return debug_options
end

struct CompileOptionsWithoutProto
    device_id::Int64
    global_device_ids::Vector{Int64}
    use_shardy_partitioner::Bool
    use_spmd_partitioning::Bool
end

function make_compile_options(;
    device_id::Int64,
    num_replicas::Int64=1,
    num_partitions::Int64=1,
    mesh_ids::Union{Vector{Int64},Nothing}=nothing,
    xla_debug_options=(;),
    xla_executable_build_options=(;),
    xla_compile_options=(;),
)
    if (
        isempty(xla_debug_options) &&
        (
            isempty(xla_executable_build_options) || (
                length(xla_executable_build_options) == 2 &&
                haskey(xla_executable_build_options, :use_shardy_partitioner) &&
                haskey(xla_executable_build_options, :use_spmd_partitioning)
            )
        ) &&
        isempty(xla_compile_options)
    )
        return CompileOptionsWithoutProto(
            device_id,
            mesh_ids === nothing ? Int64[] : mesh_ids,
            get(xla_executable_build_options, :use_shardy_partitioner, false),
            get(xla_executable_build_options, :use_spmd_partitioning, false),
        )
    end

    compile_options = get_default_compile_options()
    executable_build_options = compile_options.executable_build_options

    executable_build_options.debug_options = get_debug_options(; xla_debug_options...)
    executable_build_options.num_replicas = num_replicas
    executable_build_options.num_partitions = num_partitions

    # default overrides. can be changed by the user by passing in kwargs
    executable_build_options.allow_spmd_sharding_propagation_to_parameters = [false]
    executable_build_options.allow_spmd_sharding_propagation_to_output = [false]

    if device_id < 0
        @assert mesh_ids !== nothing
        @assert length(mesh_ids) == num_replicas * num_partitions

        computation_devices = [
            Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([
                mesh_ids[(i - 1) * num_partitions + j] for i in 1:num_replicas
            ]) for j in 1:num_partitions
        ]
        executable_build_options.device_assignment = Reactant.Proto.xla.DeviceAssignmentProto(
            Int32(num_replicas), Int32(num_partitions), computation_devices
        )
    else
        executable_build_options.device_ordinal = device_id
        executable_build_options.device_assignment = Reactant.Proto.xla.DeviceAssignmentProto(
            Int32(1),
            Int32(1),
            [Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([device_id])],
        )
    end

    for (key, val) in pairs(xla_executable_build_options)
        setproperty!(executable_build_options, key, val)
    end

    compile_options.executable_build_options = executable_build_options

    for (key, val) in pairs(xla_compile_options)
        setproperty!(compile_options, key, val)
    end

    return compile_options
end
