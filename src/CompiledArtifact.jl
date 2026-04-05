using Serialization: Serialization

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
function save_compiled_artifact(path::String, f, args...; kwargs...)
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        save_compiled_artifact(ctx, path, f, args...; kwargs...)
    end
end

function save_compiled_artifact(ctx, path::String, f, args...; kwargs...)
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

        compile_options, kwargs_inner = __get_compile_options_and_kwargs(; kwargs...)

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
        length(p) > 0 && (p[1] == :result || p[1] == :resargs)
    ]
end

function (artifact::CompiledArtifact)(args...)
    # Linearize: extract traced values from args using saved paths
    linear_args = _linearize_args(args, artifact.arg_paths)

    # Call hlo_call with the saved IR
    hlo_results = @opcall Reactant.Ops.hlo_call(artifact.ir, linear_args...)

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
        traced_val = _eval_path(args, source_path)
        for path in paths
            if path[1] == :result
                push!(result_values, traced_val)
            elseif path[1] == :resargs
                target_path = path[2:end]
                source_suffix = source_path[2:end]
                # Skip self-assignment (arg maps to itself)
                target_path == source_suffix && continue
                _update_arg!(args, path, traced_val)
            end
        end
    end

    return Tuple(result_values)
end

function _update_arg!(args, resargs_path::Tuple, traced_val)
    @assert resargs_path[1] == :resargs
    target = args[resargs_path[2]]
    # Navigate to the parent of the leaf
    for p in resargs_path[3:(end - 1)]
        target = traced_getfield(target, p)
    end
    if length(resargs_path) >= 3
        leaf = traced_getfield(target, resargs_path[end])
        Reactant.TracedUtils.set_mlir_data!(
            leaf, Reactant.TracedUtils.get_mlir_data(traced_val)
        )
    else
        # Direct arg replacement (path is just (:resargs, idx))
        Reactant.TracedUtils.set_mlir_data!(
            target, Reactant.TracedUtils.get_mlir_data(traced_val)
        )
    end
end
