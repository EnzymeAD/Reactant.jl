# TODO: document these options at some point
"""
    OptimizeCommunicationOptions

Fine-grained control over the optimization passes that rewrite ops to minimize collective
communication.
"""
@kwdef struct OptimizeCommunicationOptions
    periodic_concat::Int = 0
    rotate_comm::Int = 0
    rotate_to_pad_comm::Int = 0
    wrap_comm::Int = 0
    extend_comm::Int = 0
    dus_to_pad_manual_comp_comm::Int = 0 # 2
    dus_to_pad_comm::Int = 0
    concat_two_operands_comm::Int = 0
    concat_to_pad_comm::Int = 0
    concat_to_dus::Int = 1
    extend_to_pad_comm::Int = 0
    extend_to_pad_comm2::Int = 1
    wrap_to_pad_comm::Int = 0
    rotate_spmd::Int = 1
    wrap_to_rotate::Int = 1
    updatewithoutcorners_to_select::Int = 1
end

function Base.String(options::OptimizeCommunicationOptions)
    return (
        "optimize-communication{" *
        join(["$(f)=$(getfield(options, f))" for f in fieldnames(typeof(options))], " ") *
        "}"
    )
end

"""
    ShardyPropagationOptions

Fine-grained control over the sharding propagation pipeline. For more information on
sharding propagation, see the
[Shardy Docs](https://openxla.org/shardy/sdy_propagation_passes).

## Options

  - `keep_sharding_rules::Bool`: whether to keep existing and created op sharding rules.
  - `conservative_propagation::Bool`: whether to disallow split axes and non-divisible
    sharding axes during propagation.
  - `debug_sharding_origins::Bool`: whether to save information about the origin of a
    sharding on the MLIR module. These would be the shardings on the function inputs,
    outputs, sharding constraints and manual computations before propagation.
  - `debug_propagation_edge_sharding::Bool`: whether to save information about the edge
    source of a sharding on the MLIR module. These are what operand/result introduced a
    sharding on some op result.
  - `skip_convert_to_reshard::Bool`
  - `skip_inline::Bool`
  - `enable_insert_explicit_collectives::Bool`: whether to insert explicit collectives
    for sharding propagation. This is useful for debugging and checking the location of
    the communication ops.
"""
@kwdef struct ShardyPropagationOptions
    keep_sharding_rules::Bool = false
    conservative_propagation::Bool = false
    debug_sharding_origins::Bool = false
    debug_propagation_edge_sharding::Bool = false
    skip_convert_to_reshard::Bool = false
    skip_inline::Bool = false
    enable_insert_explicit_collectives::Bool = false
end

"""
    CompileOptions

Fine-grained control over the compilation options for the Reactant compiler.

## Controlling Optimization Passes

  - `optimization_passes`: Optimizations passes to run on the traced MLIR code. Valid types
    of values are:
    - Bool (true/false): whether to run the optimization passes or not. Defaults to `true`.
    - String: a custom string with the passes to run. The string should be a comma-separated
      list of MLIR passes. For example, `"canonicalize,enzyme-hlo-opt"`.
    - Symbol: a predefined set of passes to run. Valid options are:
       1. `:all`: Default set of optimization passes. The exact set of passes are not fixed
          and may change in future versions of Reactant. It is recommended to use this
          option for most users.
       2. `:none`: No optimization passes will be run.
       3.  Other predefined options are: `:before_kernel`, `:before_jit`, `:before_raise`,
          `:before_enzyme`, `:after_enzyme`, `:just_batch`, `:canonicalize`, `:only_enzyme`.
  - `no_nan`: If `true`, the optimization passes will assume that the function does not
    produce NaN values. This can lead to more aggressive optimizations **(and potentially
    incorrect results if the function does produce NaN values)**.
  - `all_finite`: If `true`, the optimization passes will assume that the function does not
    produce Inf or -Inf values. This can lead to more aggressive optimizations **(and
    potentially incorrect results if the function does produce Inf or -Inf values)**.
  - `transpose_propagate`: If `:up`, `stablehlo.transpose` operations will be
    propagated up the computation graph. If `:down`, they will be propagated down. Defaults
    to `:up`.
  - `reshape_propagate`: If `:up`, `stablehlo.reshape` operations will be propagated up
    the computation graph. If `:down`, they will be propagated down. Defaults to `:up`.
  - `max_constant_threshold`: If the number of elements in a constant is greater than this
    threshold (for a non-splatted constant), we will throw an error.
  - `inline`: If `true`, all functions will be inlined. (Default: `true`).

## Raising Options

  - `raise`: If `true`, the function will be compiled with the raising pass, which raises
    CUDA and KernelAbstractions kernels to HLO. Defaults to `false`, but is automatically
    activated if the inputs are sharded.
  - `raise_first`: If `true`, the raising pass will be run before the optimization passes.
    Defaults to `false`.

## Dialect Specific Options

  - `legalize_chlo_to_stablehlo`: If `true`, `chlo` dialect ops will be converted to
    `stablehlo` ops. (Default: `false`).

## Backend Specific Options

### Only for CUDA backend

  - `cudnn_hlo_optimize`: Run cuDNN specific HLO optimizations. This is only relevant for
    GPU backends and is `false` by default. **Experimental and not heavily tested.**

## Sharding Options

  - `shardy_passes`: Defaults to `:post_sdy_propagation`. Other options are:
    - `:none`: No sharding passes will be run. Shardy + MHLO shardings are handled by XLA.
    - `:post_sdy_propagation`: Runs the Shardy propagation passes. MHLO shardings are
      handled by XLA.
    - [`ShardyPropagationOptions`](@ref): Custom sharding propagation options.
      MHLO shardings are handled by XLA.
    - `:to_mhlo_shardings`: Runs the Shardy propagation passes and then exports the
      shardings to MHLO. All passes are run via MLIR pass pipeline and don't involve XLA.
  - `optimize_then_pad`: If `true`, the function will be optimized before padding (for
    non-divisible sharding axes) is applied. Defaults to `true`. _(Only for Sharded Inputs)_
  - `optimize_communications`: If `true`, additional passes for optimizing communication
    in sharded computations will be run. Defaults to `true`. _(Only for Sharded Inputs)_

## Julia Codegen Options

  - `donated_args`: If `:auto`, the function will automatically donate the arguments that
    are not preserved in the function body. If `:none`, no arguments will be donated.
    Defaults to `:auto`.
  - `assert_nonallocating`: If `true`, we make sure that no new buffers are
    returned by the function. Any buffer returned must be donated from the inputs. Defaults
    to `false`.
  - `sync`: Reactant computations are asynchronous by default. If `true`, the computation
    will be executed synchronously, blocking till the computation is complete. This is
    recommended when benchmarking.

# Extended Help

## Private Options

!!! warning

    These options are not part of the public API and are subject to change without any
    notice or deprecation cycle.

  - `disable_scatter_gather_optimization_passes`: Disables the scatter-gather
    optimization passes. (Default: `false`).
  - `disable_pad_optimization_passes`: Disables the pad optimization passes. This is
    `false` by default.
  - `disable_licm_optimization_passes`: Disables the Loop Invariant Code Motion (LICM)
    optimization passes. (Default: `false`).
  - `disable_reduce_slice_fusion_passes`: Disables fusion of slice elementwise and reduce
    operations. (Default `false`).
  - `disable_slice_to_batch_passes`: Disables the slice to batch fusion optimization passes.
    (Default: `true`). _(Note that this is generally an expensive pass to run)_
  - `disable_concat_to_batch_passes`: Disables concatenate to batch fusion passes.
    (Default: `false`).
  - `disable_loop_raising_passes`: Disables raising passes for `stablehlo.while`.
    (Default: `false`).
  - `disable_structured_tensors_detection_passes`: Disables structured tensors detection
    passes. (Default `true`).
  - `disable_structured_tensors_passes`: Disables structured tensors optimization passes.
    (Default `false`).
"""
struct CompileOptions
    optimization_passes::Union{Symbol,String}
    no_nan::Bool
    all_finite::Bool
    inline::Bool
    transpose_propagate::Symbol
    reshape_propagate::Symbol
    max_constant_threshold::Int
    # Raising options
    raise::Union{Bool,String}
    raise_first::Bool
    # dialect specific options
    legalize_chlo_to_stablehlo::Bool
    # backend specific options
    cudnn_hlo_optimize::Bool
    # sharding options
    shardy_passes::Union{Symbol,ShardyPropagationOptions}
    optimize_then_pad::Bool
    optimize_communications::Union{Bool,OptimizeCommunicationOptions}
    # julia codegen options
    assert_nonallocating::Bool
    donated_args::Symbol
    sync::Bool
    ## private options for ablation studies
    disable_scatter_gather_optimization_passes::Bool
    disable_pad_optimization_passes::Bool
    disable_licm_optimization_passes::Bool
    disable_reduce_slice_fusion_passes::Bool
    disable_slice_to_batch_passes::Bool
    disable_concat_to_batch_passes::Bool
    disable_loop_raising_passes::Bool
    disable_structured_tensors_detection_passes::Bool
    disable_structured_tensors_passes::Bool
end

function CompileOptions(;
    optimization_passes::Union{Bool,Symbol,String}=:all,
    no_nan::Bool=false,
    all_finite::Bool=false,
    inline::Bool=true,
    transpose_propagate::Symbol=:up,
    reshape_propagate::Symbol=:up,
    max_constant_threshold::Int=1024,
    raise::Union{Bool,String}=false,
    raise_first::Bool=false,
    legalize_chlo_to_stablehlo::Bool=false,
    cudnn_hlo_optimize::Bool=false,
    shardy_passes::Union{Symbol,ShardyPropagationOptions}=:to_mhlo_shardings,
    optimize_then_pad::Bool=true,
    optimize_communications::Union{Bool,OptimizeCommunicationOptions}=true,
    assert_nonallocating::Bool=false,
    donated_args::Symbol=:auto,
    sync::Bool=false,
    disable_scatter_gather_optimization_passes::Bool=false,
    disable_pad_optimization_passes::Bool=false,
    disable_licm_optimization_passes::Bool=false,
    disable_reduce_slice_fusion_passes::Bool=false,
    disable_slice_to_batch_passes::Bool=true, # expensive + introduces all-to-all in GB25
    disable_concat_to_batch_passes::Bool=false,
    disable_loop_raising_passes::Bool=false,
    disable_structured_tensors_detection_passes::Bool=true,  # missing optimization passes currently
    disable_structured_tensors_passes::Bool=false,
)
    optimization_passes isa Bool &&
        (optimization_passes = ifelse(optimization_passes, :all, :none))

    if optimization_passes isa Symbol
        @assert optimization_passes in [
            :all,
            :before_kernel,
            :before_jit,
            :before_raise,
            :no_enzyme,
            :only_enzyme,
            :after_enzyme,
            :before_enzyme,
            :canonicalize,
            :just_batch,
            :none,
        ]
    end

    @assert transpose_propagate in [:up, :down, :none]
    @assert reshape_propagate in [:up, :down, :none]

    if shardy_passes isa Symbol
        @assert shardy_passes in [:none, :to_mhlo_shardings, :post_sdy_propagation]
    end

    return CompileOptions(
        optimization_passes,
        no_nan,
        all_finite,
        inline,
        transpose_propagate,
        reshape_propagate,
        max_constant_threshold,
        raise,
        raise_first,
        legalize_chlo_to_stablehlo,
        cudnn_hlo_optimize,
        shardy_passes,
        optimize_then_pad,
        optimize_communications,
        assert_nonallocating,
        donated_args,
        sync,
        disable_scatter_gather_optimization_passes,
        disable_pad_optimization_passes,
        disable_licm_optimization_passes,
        disable_reduce_slice_fusion_passes,
        disable_slice_to_batch_passes,
        disable_concat_to_batch_passes,
        disable_loop_raising_passes,
        disable_structured_tensors_detection_passes,
        disable_structured_tensors_passes,
    )
end

function __compile_options_from_kwags(;
    compile_options::Union{Missing,CompileOptions}=missing,
    optimize::Union{Bool,Symbol,String}=true,
    kwargs...,
)
    compile_options isa CompileOptions && return compile_options
    return CompileOptions(; optimization_passes=optimize, kwargs...)
end

function __reverse_propagation(sym::Symbol)
    sym == :up && return :down
    sym === :down && return :up
    sym == :none && return :none
    return error("Invalid value: $sym. Expected :up or :down or :none")
end

function __compile_options_with_reversed_propagation(compile_options::CompileOptions)
    return CompileOptions(
        compile_options.optimization_passes,
        compile_options.no_nan,
        compile_options.all_finite,
        compile_options.inline,
        __reverse_propagation(compile_options.transpose_propagate),
        __reverse_propagation(compile_options.reshape_propagate),
        compile_options.max_constant_threshold,
        compile_options.raise,
        compile_options.raise_first,
        compile_options.legalize_chlo_to_stablehlo,
        compile_options.cudnn_hlo_optimize,
        compile_options.shardy_passes,
        compile_options.optimize_then_pad,
        compile_options.optimize_communications,
        compile_options.assert_nonallocating,
        compile_options.donated_args,
        compile_options.sync,
        compile_options.disable_scatter_gather_optimization_passes,
        compile_options.disable_pad_optimization_passes,
        compile_options.disable_licm_optimization_passes,
        compile_options.disable_reduce_slice_fusion_passes,
        compile_options.disable_slice_to_batch_passes,
        compile_options.disable_concat_to_batch_passes,
        compile_options.disable_loop_raising_passes,
        compile_options.disable_structured_tensors_detection_passes,
        compile_options.disable_structured_tensors_passes,
    )
end

function __compile_options_with_updated_sync(compile_options::CompileOptions, sync::Bool)
    if compile_options.sync == sync
        return compile_options
    end
    return CompileOptions(
        compile_options.optimization_passes,
        compile_options.no_nan,
        compile_options.all_finite,
        compile_options.inline,
        compile_options.transpose_propagate,
        compile_options.reshape_propagate,
        compile_options.max_constant_threshold,
        compile_options.raise,
        compile_options.raise_first,
        compile_options.legalize_chlo_to_stablehlo,
        compile_options.cudnn_hlo_optimize,
        compile_options.shardy_passes,
        compile_options.optimize_then_pad,
        compile_options.optimize_communications,
        compile_options.assert_nonallocating,
        compile_options.donated_args,
        sync,
        compile_options.disable_scatter_gather_optimization_passes,
        compile_options.disable_pad_optimization_passes,
        compile_options.disable_licm_optimization_passes,
        compile_options.disable_reduce_slice_fusion_passes,
        compile_options.disable_slice_to_batch_passes,
        compile_options.disable_concat_to_batch_passes,
        compile_options.disable_loop_raising_passes,
        compile_options.disable_structured_tensors_detection_passes,
        compile_options.disable_structured_tensors_passes,
    )
end

"""
    DefaultXLACompileOptions(;
        donated_args=:auto, sync=false, optimize_then_pad=true, assert_nonallocating=false
    )

Runs specific Enzyme-JAX passes to ensure that the generated code is compatible with
XLA compilation. For the documentation of the allowed kwargs see [`CompileOptions`](@ref).

!!! warning

    This is mostly a benchmarking option, and the default [`CompileOptions`](@ref) is almost
    certainly a better option.
"""
function DefaultXLACompileOptions(;
    donated_args=:auto, sync=false, optimize_then_pad=true, assert_nonallocating=false
)
    return CompileOptions(;
        optimization_passes=:only_enzyme,
        inline=false,
        donated_args,
        sync,
        optimize_then_pad,
        assert_nonallocating,
        optimize_communications=false,
    )
end
