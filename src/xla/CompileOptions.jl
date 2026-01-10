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

    @set! debug_options.xla_gpu_cuda_data_dir = CUDA_DATA_DIR[]
    @set! debug_options.xla_enable_enzyme_comms_opt = true
    @set! debug_options.xla_gpu_experimental_use_raft_select_k = true

    if Reactant.PersistentCompileCache.kernel_cache_enabled()
        @set! debug_options.xla_gpu_kernel_cache_file = Reactant.PersistentCompileCache.get_kernel_cache_path()
        @set! debug_options.xla_gpu_enable_llvm_module_compilation_parallelism = true
    end
    if Reactant.PersistentCompileCache.autotune_cache_enabled()
        @set! debug_options.xla_gpu_per_fusion_autotune_cache_dir = Reactant.PersistentCompileCache.get_autotune_cache_directory()
        if Reactant.Distributed.local_rank() <= 0
            @set! debug_options.xla_gpu_experimental_autotune_cache_mode =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UPDATE
        else
            @set! debug_options.xla_gpu_experimental_autotune_cache_mode =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_READ
        end
    end

    for (key, value) in kwargs
        debug_options = Setfield.set(debug_options, Setfield.PropertyLens{key}(), value)
    end

    return debug_options
end

function make_compile_options(;
    device_id::Int64,
    num_replicas::Int64=1,
    num_partitions::Int64=1,
    use_shardy_partitioner::Bool=true,
    use_spmd_partitioning::Bool=true,
    mesh_ids::Union{Vector{Int64},Nothing}=nothing,
    xla_debug_options=(;),
    xla_executable_build_options=(;),
    xla_compile_options=(;),
)
    compile_options = get_default_compile_options()
    executable_build_options = compile_options.executable_build_options

    @set! executable_build_options.debug_options = get_debug_options(; xla_debug_options...)
    @set! executable_build_options.num_replicas = num_replicas
    @set! executable_build_options.num_partitions = num_partitions
    @set! executable_build_options.use_spmd_partitioning = use_spmd_partitioning
    @set! executable_build_options.use_shardy_partitioner = use_shardy_partitioner
    @set! executable_build_options.allow_spmd_sharding_propagation_to_parameters = [false]
    @set! executable_build_options.allow_spmd_sharding_propagation_to_output = [false]

    if device_id < 0
        @assert !isnothing(mesh_ids)
        @assert length(mesh_ids) == num_replicas * num_partitions

        computation_devices = [
            Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([
                mesh_ids[(i - 1) * num_partitions + j] for i in 1:num_replicas
            ]) for j in 1:num_partitions
        ]
        @set! executable_build_options.device_assignment = Reactant.Proto.xla.DeviceAssignmentProto(
            Int32(num_replicas), Int32(num_partitions), computation_devices
        )
    else
        @set! executable_build_options.device_ordinal = device_id
        @set! executable_build_options.device_assignment = Reactant.Proto.xla.DeviceAssignmentProto(
            Int32(1),
            Int32(1),
            [Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([device_id])],
        )
    end

    for (key, val) in xla_executable_build_options
        executable_build_options = Setfield.set(
            executable_build_options, Setfield.PropertyLens{key}(), val
        )
    end

    @set! compile_options.executable_build_options = executable_build_options

    for (key, val) in xla_compile_options
        compile_options = Setfield.set(compile_options, Setfield.PropertyLens{key}(), val)
    end

    return compile_options
end
