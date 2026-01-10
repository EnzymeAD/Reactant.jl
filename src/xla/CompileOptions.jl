function get_debug_options(; kwargs...)
    opts = Dict{Symbol,Any}(
        :xla_gpu_cuda_data_dir => CUDA_DATA_DIR[],
        :xla_enable_enzyme_comms_opt => true,
        :xla_gpu_experimental_use_raft_select_k => true,
    )
    if Reactant.PersistentCompileCache.kernel_cache_enabled()
        opts[:xla_gpu_kernel_cache_file] = Reactant.PersistentCompileCache.get_kernel_cache_path()
        opts[:xla_gpu_enable_llvm_module_compilation_parallelism] = true
    end
    if Reactant.PersistentCompileCache.autotune_cache_enabled()
        opts[:xla_gpu_per_fusion_autotune_cache_dir] = Reactant.PersistentCompileCache.get_autotune_cache_directory()
        if Reactant.Distributed.local_rank() <= 0
            opts[:xla_gpu_experimental_autotune_cache_mode] =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_UPDATE
        else
            opts[:xla_gpu_experimental_autotune_cache_mode] =
                Reactant.Proto.xla.var"DebugOptions.AutotuneCacheMode".AUTOTUNE_CACHE_MODE_READ
        end
    end

    return Reactant.Proto.xla.DebugOptions(; kwargs..., opts...)
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
    debug_options = get_debug_options(; xla_debug_options...)

    exec_opts_dict = Dict{Symbol,Any}(
        :debug_options => debug_options,
        :num_replicas => num_replicas,
        :num_partitions => num_partitions,
        :use_spmd_partitioning => use_spmd_partitioning,
        :use_shardy_partitioner => use_shardy_partitioner,
        :allow_spmd_sharding_propagation_to_parameters => [false],
        :allow_spmd_sharding_propagation_to_output => [false],
    )

    if device_id < 0
        @assert !isnothing(mesh_ids)
        @assert length(mesh_ids) == num_replicas * num_partitions

        computation_devices = [
            Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([
                mesh_ids[(i - 1) * num_partitions + j] for i in 1:num_replicas
            ]) for j in 1:num_partitions
        ]
        exec_opts_dict[:device_assignment] = Reactant.Proto.xla.DeviceAssignmentProto(;
            replica_count=Int32(num_replicas),
            computation_count=Int32(num_partitions),
            computation_devices=computation_devices,
        )
    else
        exec_opts_dict[:device_ordinal] = device_id
        exec_opts_dict[:device_assignment] = Reactant.Proto.xla.DeviceAssignmentProto(;
            replica_count=Int32(1),
            computation_count=Int32(1),
            computation_devices=[
                Reactant.Proto.xla.var"DeviceAssignmentProto.ComputationDevice"([device_id])
            ],
        )
    end

    exec_opts = Reactant.Proto.xla.ExecutableBuildOptionsProto(;
        xla_executable_build_options..., exec_opts_dict...
    )
    compile_opts = Reactant.Proto.xla.CompileOptionsProto(;
        xla_compile_options..., executable_build_options=exec_opts
    )

    return compile_opts
end
