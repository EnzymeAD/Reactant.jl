using Serialization: Serialization
using Reactant: Ops, Ops.hlo_call

struct CompiledArtifact
    # The optimized MLIR IR as a string (printed with debug locations)
    ir::String

    # Arg paths: one per linear/block argument, e.g. (:args, 1, :field)
    arg_paths::Vector{Tuple}

    # Result paths: one per linear result (preserved args excluded from IR returns).
    # Each entry is a vector of paths because a single linear result can appear at
    # multiple locations (e.g. both :result and :resargs).
    result_paths::Vector{Vector{Tuple}}

    # Preserved args: (result_paths, block_arg_0based_index).
    # These are results that are block-arg pass-throughs (not returned by the IR).
    preserved_args::Vector{Tuple{Vector{Tuple},Int}}

    # Donated args mask
    donated_args_mask::Vector{Bool}
end

function CompiledArtifact(path_to_file::AbstractString)
    return Serialization.deserialize(path_to_file)::CompiledArtifact
end

"""
    save_compiled_artifact(path::String, f, args...; kwargs...)

Compile `f(args...)` through the MLIR pipeline and serialize the optimized IR together
with argument/result path metadata to `path` using Julia's `Serialization` package.
"""
function save_compiled_artifact(path::String, f, args...; raise=true, kwargs...)
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        save_compiled_artifact(ctx, path, f, args...; raise, kwargs...)
    end
end

function save_compiled_artifact(ctx, path::String, f, args...; raise=true, kwargs...)
    raise == false && error("save_compiled_artifact requires raise=true for portable IR")

    client = XLA.default_backend()
    backend = XLA.platform_name(client)

    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    MLIR.IR.activate(ctx)
    try
        mod = MLIR.IR.Module(MLIR.IR.Location())

        compile_options, kwargs_inner = __get_compile_options_and_kwargs(; raise, kwargs...)

        mlir_fn_res = compile_mlir!(
            mod,
            f,
            args,
            compile_options;
            backend,
            runtime=XLA.runtime(client),
            client,
            kwargs_inner...,
        )

        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )
        run_pass_pipeline!(mod, "drop-unsupported-attributes", "drop_enzymexla_attributes")

        # Extract IR string with debug locations
        iobuffer = IOBuffer()
        show(IOContext(iobuffer, :debug => true), mod)
        ir = String(take!(iobuffer))

        # Extract arg paths
        arg_paths = Tuple[
            Reactant.TracedUtils.get_idx(arg, :args) for arg in mlir_fn_res.linear_args
        ]

        # Extract result paths (filtered to :result / :resargs)
        result_paths = Vector{Tuple}[
            _collect_result_paths(res) for res in mlir_fn_res.linear_results
        ]

        # Extract preserved args
        preserved = Tuple{Vector{Tuple},Int}[
            (_collect_result_paths(traced_result), block_arg_idx) for
            (traced_result, block_arg_idx) in mlir_fn_res.preserved_args
        ]

        artifact = CompiledArtifact(
            ir,
            arg_paths,
            result_paths,
            preserved,
            collect(Bool, mlir_fn_res.donated_args_mask),
        )

        open(path, "w") do io
            Serialization.serialize(io, artifact)
        end

        return artifact
    finally
        MLIR.IR.deactivate(ctx)
    end
end

function _collect_result_paths(traced_val)
    return Tuple[
        p for p in Reactant.TracedUtils.get_paths(traced_val) if
        length(p) > 0 && (p[1] === :result || p[1] == :resargs || p[1] === :args)
    ]
end

function (artifact::CompiledArtifact)(args...)
    # Linearize: extract traced values from args using saved paths
    linear_args = _linearize_args(args, artifact.arg_paths)

    # Call hlo_call with the saved IR
    hlo_results = Ops.@opcall hlo_call(artifact.ir, linear_args...)

    # Delinearize: update args and collect result-path outputs
    return _delinearize_results(args, hlo_results, artifact)
end

function _linearize_args(args, arg_paths::Vector{Tuple})
    n = length(arg_paths)
    linear_args = Vector{Any}(undef, n)
    for i in 1:n
        linear_args[i] = _eval_path(args, @inbounds arg_paths[i])
    end
    return linear_args
end

function _eval_path(args, path::Tuple)
    @assert length(path) >= 2 && path[1] === :args "Expected path starting with :args, got $path"
    val = args[path[2]]
    for p in path[3:end]
        val = traced_getfield(val, p)
    end
    return val
end

function _delinearize_results(args, hlo_results, artifact::CompiledArtifact)
    result_values = []

    # Process hlo_call results (non-preserved)
    for (i, paths) in enumerate(artifact.result_paths)
        traced_val = hlo_results[i]
        for path in paths
            if path[1] == :result
                push!(result_values, traced_val)
            elseif path[1] == :resargs
                _update_arg!(args, path, traced_val)
            end
        end
    end

    # Process preserved args (block arg pass-throughs)
    for (paths, block_arg_idx) in artifact.preserved_args
        # The traced value is the input arg itself (it was a pass-through)
        source_path = artifact.arg_paths[block_arg_idx + 1]
        val = _eval_path(args, source_path)
        for path in paths
            if path[1] == :result
                push!(result_values, val)
            elseif path[1] == :resargs
                target_path = path[2:end]
                source_suffix = source_path[2:end]
                # Skip self-assignment (arg maps to itself)
                target_path == source_suffix && continue
                _update_arg!(args, path, copy(val.data)) # TODO: copy here is wrong, do the same as in Compiler.jl
            end
        end
    end

    return Tuple(result_values)
end

"""
    SerializedExecutableArtifact

Like `CompiledArtifact` but additionally contains serialized XLA executable bytes,
enabling direct execution without recompilation. Created via
[`save_compiled_executable`](@ref) with a topology argument.
"""
struct SerializedExecutableArtifact
    artifact::CompiledArtifact
    executable_bytes::Vector{UInt8}
    num_outputs::Int64
    num_parameters::Int64
    is_sharded::Bool
    num_replicas::Int64
    num_partitions::Int64
    # Sharding info for each linear result (from the concrete_result at compile time).
    # Each entry is the ShardInfo of the corresponding result array, or nothing if unsharded.
    result_shardings::Vector{Any}
    # Sharding info for preserved args (same indexing as artifact.preserved_args)
    preserved_shardings::Vector{Any}
end

"""
    save_compiled_executable(path, f, args...; topology, kwargs...)

Compile `f(args...)` and additionally compile the optimized IR against a mock GPU
`topology` (an `XLA.IFRT.GpuTopology`). Saves both the MLIR IR metadata and the
serialized XLA executable to `path`.

The saved artifact can be loaded on real hardware with
[`load_compiled_executable`](@ref) to skip recompilation entirely.
"""
function save_compiled_executable(path::String, f, args...; topology, raise=true, kwargs...)
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        save_compiled_executable(ctx, path, f, args...; topology, raise, kwargs...)
    end
end

function save_compiled_executable(
    ctx, path::String, f, args...; topology, raise=true, kwargs...
)
    client = XLA.default_backend()
    backend = XLA.platform_name(client)

    raise || error("save_compiled_executable requires raise=true for portable IR")
    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    MLIR.IR.activate(ctx)
    try
        mod = MLIR.IR.Module(MLIR.IR.Location())

        compile_options, kwargs_inner = __get_compile_options_and_kwargs(; raise, kwargs...)

        mlir_fn_res = compile_mlir!(
            mod,
            f,
            args,
            compile_options;
            backend,
            runtime=XLA.runtime(client),
            client,
            kwargs_inner...,
        )

        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )
        run_pass_pipeline!(mod, "drop-unsupported-attributes", "drop_enzymexla_attributes")

        # Build the base CompiledArtifact (IR + paths)
        iobuffer = IOBuffer()
        show(IOContext(iobuffer, :debug => true), mod)
        ir = String(take!(iobuffer))

        arg_paths = Tuple[
            Reactant.TracedUtils.get_idx(arg, :args) for arg in mlir_fn_res.linear_args
        ]
        result_paths = Vector{Tuple}[
            _collect_result_paths(res) for res in mlir_fn_res.linear_results
        ]
        preserved = Tuple{Vector{Tuple},Int}[
            (_collect_result_paths(traced_result), block_arg_idx) for
            (traced_result, block_arg_idx) in mlir_fn_res.preserved_args
        ]
        artifact = CompiledArtifact(
            ir,
            arg_paths,
            result_paths,
            preserved,
            collect(Bool, mlir_fn_res.donated_args_mask),
        )

        # Compile against the mock topology
        if mlir_fn_res.is_sharded
            xla_compile_options = XLA.make_compile_options(;
                device_id=Int64(-1),
                num_replicas=mlir_fn_res.num_replicas,
                num_partitions=mlir_fn_res.num_partitions,
                mesh_ids=nothing,
                xla_compile_options=compile_options.xla_compile_options,
                xla_debug_options=merge(
                    (; xla_gpu_experimental_aot_compiled_thunks=true),
                    compile_options.xla_debug_options,
                ),
                xla_executable_build_options=merge(
                    (; use_shardy_partitioner=true, use_spmd_partitioning=true),
                    compile_options.xla_executable_build_options,
                ),
            )
        else
            xla_compile_options = XLA.make_compile_options(;
                device_id=Int64(0),
                num_replicas=mlir_fn_res.num_replicas,
                num_partitions=mlir_fn_res.num_partitions,
                xla_compile_options=compile_options.xla_compile_options,
                xla_debug_options=compile_options.xla_debug_options,
                xla_executable_build_options=merge(
                    (; use_shardy_partitioner=false, use_spmd_partitioning=false),
                    compile_options.xla_executable_build_options,
                ),
            )
        end

        compil_client = XLA.client("gpu")
        exec = XLA.IFRT.compile_to_executable(
            compil_client, mod, topology; compile_options=xla_compile_options
        )
        executable_bytes = XLA.IFRT.serialize_executable(exec)

        # For preserved args, extract sharding from the corresponding input arg
        preserved_shardings = Any[]
        for (_, block_arg_idx) in mlir_fn_res.preserved_args
            arg = mlir_fn_res.linear_args[block_arg_idx + 1]
            arg_path = Reactant.TracedUtils.get_idx(arg, :args)
            val = args[arg_path[2]]
            for p in arg_path[3:end]
                val = traced_getfield(val, p)
            end
            if val isa Reactant.AbstractConcreteArray
                push!(preserved_shardings, val.sharding)
            else
                push!(preserved_shardings, Reactant.Sharding.NoShardInfo())
            end
        end

        result_shardings = mlir_fn_res.result_shardings

        serialized = SerializedExecutableArtifact(
            artifact,
            executable_bytes,
            Int64(length(mlir_fn_res.linear_results)),
            Int64(length(mlir_fn_res.linear_args)),
            mlir_fn_res.is_sharded,
            Int64(mlir_fn_res.num_replicas),
            Int64(mlir_fn_res.num_partitions),
            result_shardings,
            preserved_shardings,
        )

        open(path, "w") do io
            Serialization.serialize(io, serialized)
        end

        return serialized
    finally
        MLIR.IR.deactivate(ctx)
    end
end

"""
    load_compiled_executable(path, args...; client=nothing)

Load a `SerializedExecutableArtifact` from `path`, deserialize the XLA executable
onto `client`'s devices, and execute it with `args`. Skips recompilation entirely.

Returns a flat tuple of concrete results.
"""
function load_compiled_executable(path::String, args...; client=nothing)
    serialized = open(path, "r") do io
        Serialization.deserialize(io)::SerializedExecutableArtifact
    end
    return load_compiled_executable(serialized, args...; client)
end

function load_compiled_executable(
    serialized::SerializedExecutableArtifact, args...; client=nothing, kwargs...
)
    client = client !== nothing ? client : XLA.default_backend()

    compile_options, _ = __get_compile_options_and_kwargs(; kwargs...)

    xla_compile_options = if serialized.is_sharded
        global_device_ids = collect(Int64, 0:1)
        XLA.make_compile_options(;
            device_id=Int64(-1),
            num_replicas=serialized.num_replicas,
            num_partitions=serialized.num_partitions,
            mesh_ids=global_device_ids,
            xla_compile_options=compile_options.xla_compile_options,
            xla_debug_options=compile_options.xla_debug_options,
            xla_executable_build_options=merge(
                (; use_shardy_partitioner=true, use_spmd_partitioning=true),
                compile_options.xla_executable_build_options,
            ),
        )
    else
        XLA.make_compile_options(;
            device_id=Int64(0),
            num_replicas=serialized.num_replicas,
            num_partitions=serialized.num_partitions,
            xla_compile_options=compile_options.xla_compile_options,
            xla_debug_options=compile_options.xla_debug_options,
            xla_executable_build_options=merge(
                (; use_shardy_partitioner=true, use_spmd_partitioning=true),
                compile_options.xla_executable_build_options,
            ),
        )
    end

    # Deserialize and load the executable
    exec = XLA.IFRT.deserialize_and_load(
        client,
        serialized.executable_bytes;
        num_outputs=serialized.num_outputs,
        num_parameters=serialized.num_parameters,
        is_sharded=serialized.is_sharded,
        num_replicas=serialized.num_replicas,
        num_partitions=serialized.num_partitions,
        compile_options=xla_compile_options,
    )

    # Linearize args: extract XLA buffers
    artifact = serialized.artifact
    linear_args = _linearize_concrete_args(args, artifact.arg_paths)

    @info "linae" linear_args

    for (arr, d) in zip(linear_args, artifact.donated_args_mask)
        if d
            #Reactant.mark_donated!(arr)
        end
    end

    linear_ptrs = Tuple((arr.data.buffer.buffer for arr in linear_args))

    num_outs = Int(serialized.num_outputs)
    donated_arr = Tuple(UInt8(d) for d in artifact.donated_args_mask)

    output_arr = XLA.execute(exec, linear_ptrs, donated_arr, Val(num_outs))

    @show output_arr

    # Delinearize: wrap results back
    return _delinearize_results(args, output_arr, artifact)
end

function _linearize_concrete_args(args, arg_paths::Vector{Tuple})
    n = length(arg_paths)
    linear_args = Vector{Any}(undef, n)
    for i in 1:n
        path = @inbounds arg_paths[i]
        @assert length(path) >= 2 && path[1] === :args
        val = args[path[2]]
        for p in path[3:end]
            val = traced_getfield(val, p)
        end
        XLA.synced_buffer(val.data)
        linear_args[i] = val
    end
    return linear_args
end

"""
    _per_shard_to_global_shape(sharding, per_shard_shape)

Reconstruct the global array shape from the per-shard shape and the sharding spec.
In SPMD, each device holds a slice of the global array. The global shape along each
dimension is: per_shard_dim * (product of mesh axis sizes that shard this dim).
"""
function _per_shard_to_global_shape(
    sharding::Reactant.Sharding.NamedSharding, per_shard_shape
)
    mesh = sharding.mesh
    axis_name_to_size = Dict{Symbol,Int}(zip(mesh.axis_names, mesh.axis_sizes))
    return ntuple(length(per_shard_shape)) do d
        pspec = sharding.partition_spec[d]
        factor = 1
        for axis in pspec
            axis === nothing && continue
            factor *= get(axis_name_to_size, axis, 1)
        end
        per_shard_shape[d] * factor
    end
end

function _per_shard_to_global_shape(::Reactant.Sharding.Replicated, per_shard_shape)
    return per_shard_shape
end

function _per_shard_to_global_shape(::Reactant.Sharding.NoSharding, per_shard_shape)
    return per_shard_shape
end

function _sharding_to_shardinfo(sharding, concrete_result, args, paths)
    # For NoSharding/Replicated without a mesh, just return NoShardInfo
    if sharding isa Reactant.Sharding.NoSharding
        return Reactant.Sharding.NoShardInfo()
    end
    # Navigate to the concrete array to get its existing ShardInfo
    for path in paths
        if path[1] == :result
            val = concrete_result
            for p in path[2:end]
                val = traced_getfield(val, p)
            end
            if val isa Reactant.AbstractConcreteArray
                return val.sharding
            end
        elseif path[1] == :resargs
            val = args[path[2]]
            for p in path[3:end]
                val = traced_getfield(val, p)
            end
            if val isa Reactant.AbstractConcreteArray
                return val.sharding
            end
        end
    end
    return Reactant.Sharding.NoShardInfo()
end

function _extract_shardinfo(concrete_result, args, paths)
    for path in paths
        if path[1] == :result
            val = concrete_result
            for p in path[2:end]
                val = traced_getfield(val, p)
            end
            if val isa Reactant.AbstractConcreteArray
                return val.sharding
            end
        elseif path[1] == :resargs
            val = args[path[2]]
            for p in path[3:end]
                val = traced_getfield(val, p)
            end
            if val isa Reactant.AbstractConcreteArray
                return val.sharding
            end
        end
    end
    return Reactant.Sharding.NoShardInfo()
end

function _update_arg!(args, resargs_path::Tuple, val)
    @assert resargs_path[1] == :resargs
    target = args[resargs_path[2]]
    # Navigate to the parent of the leaf
    for p in resargs_path[3:end]
        target = traced_getfield(target, p)
    end
    @info "setfield!" target resargs_path val
    return setfield!(target, :data, val)
end
