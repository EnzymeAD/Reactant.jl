# This file contains the MLIR optimization pass logic.

using ..Reactant:
    Reactant, MLIR, OptimizeCommunicationOptions, ShardyPropagationOptions, CompileOptions

const BFLOAT16_COMPILE_TYPE = Ref{DataType}(Float32)
const DEBUG_KERNEL = Ref{Bool}(false)
const DUMP_LLVMIR = Ref{Bool}(false)
const DUMP_FAILED_LOCKSTEP = Ref{Bool}(false)
const OpenMP = Ref{Bool}(true)
const SROA_ATTRIBUTOR = Ref{Bool}(true)
const DEBUG_PROBPROG_DUMP_VALUE = Ref(false)
const DEBUG_PROBPROG_DISABLE_OPT = Ref(true)

const WHILE_CONCAT = Ref(false)
const DUS_TO_CONCAT = Ref(false)
const SUM_TO_REDUCEWINDOW = Ref(false)
const SUM_TO_CONV = Ref(false)
const AGGRESSIVE_SUM_TO_CONV = Ref(false)
const AGGRESSIVE_PROPAGATION = Ref(false)
const DUS_SLICE_SIMPLIFY = Ref(true)
const CONCATS_TO_DUS = Ref(false)
const WHILE_UNROLL_THRESHOLD = Ref(4)

function _propagate_direction(sym::Symbol)
    sym === :up && return MLIR.API.ENZYMEXLA_PROPAGATE_UP
    sym === :down && return MLIR.API.ENZYMEXLA_PROPAGATE_DOWN
    return MLIR.API.ENZYMEXLA_PROPAGATE_NONE
end

# Optimization passes via transform dialect
function optimization_passes(
    compile_options::CompileOptions;
    sroa::Bool=false,
    dus_to_concat::Bool=false,
    recognize_comms::Bool=true,
    lower_comms::Bool=true,
    backend::String="gpu",
    is_sharded::Bool=false,
    raise_shlo_to_blas_lapack::Bool=true,
    self_to_convolution::Bool=false,
)
    options = MLIR.API.EnzymeXLATransformPassesOptions(
        compile_options.max_constant_threshold,          # max_constant_threshold
        WHILE_UNROLL_THRESHOLD[],                        # while_unroll_threshold
        _propagate_direction(compile_options.reshape_propagate),   # reshape_propagate
        _propagate_direction(compile_options.transpose_propagate), # transpose_propagate
        compile_options.no_nan,                          # no_nan
        compile_options.all_finite,                      # all_finite
        dus_to_concat,                                   # dus_to_concat
        DUS_SLICE_SIMPLIFY[],                            # dus_slice_simplify
        SUM_TO_REDUCEWINDOW[],                           # sum_to_reducewindow
        SUM_TO_CONV[],                                   # sum_to_conv
        AGGRESSIVE_SUM_TO_CONV[],                        # aggressive_sum_to_conv
        WHILE_CONCAT[],                                  # while_concat
        AGGRESSIVE_PROPAGATION[],                        # aggressive_propagation
        is_sharded,                                      # is_sharded
        raise_shlo_to_blas_lapack,                       # raise_shlo_to_blas_lapack
        recognize_comms,                                 # recognize_comms
        lower_comms,                                     # lower_comms
        self_to_convolution,                             # enable_self_to_convolution_like_passes
        !compile_options.disable_structured_tensors_detection_passes, # enable_structured_tensors_detection_passes
        !compile_options.disable_structured_tensors_passes,          # enable_structured_tensors_passes
        !compile_options.disable_scatter_gather_optimization_passes, # enable_scatter_gather_optimization_passes
        !compile_options.disable_slice_to_batch_passes,              # enable_slice_to_batch_passes
        !compile_options.disable_reduce_slice_fusion_passes,         # enable_reduce_slice_fusion_passes
        !compile_options.disable_concat_to_batch_passes,             # enable_concat_to_batch_passes
        !compile_options.disable_loop_raising_passes,                # enable_loop_raising_passes
        !compile_options.disable_licm_optimization_passes,           # enable_licm_optimization_passes
        !compile_options.disable_pad_optimization_passes,            # enable_pad_optimization_passes
    )

    main_passes_ptr = Ref{Cstring}()
    lower_passes_ptr = Ref{Cstring}()
    GC.@preserve options begin
        MLIR.API.enzymexlaGetTransformPassesList(
            Ref(options), main_passes_ptr, lower_passes_ptr
        )
    end
    main_passes_str = unsafe_string(main_passes_ptr[])
    lower_passes_str = unsafe_string(lower_passes_ptr[])
    MLIR.API.enzymexlaFreeTransformPassesList(main_passes_ptr[])
    MLIR.API.enzymexlaFreeTransformPassesList(lower_passes_ptr[])
    main_passes_str = replace(main_passes_str, "convert_mul_convert;" => "")
    lower_passes_str = replace(lower_passes_str, "convert_mul_convert;" => "")

    main_passes_str = replace(main_passes_str, "associative_binary_op_reordering<1>;" => "")
    lower_passes_str = replace(
        lower_passes_str, "associative_binary_op_reordering<1>;" => ""
    )

    transform_passes = join(
        [
            "enzyme-hlo-generate-td{patterns=" * main_passes_str * "}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ],
        ",",
    )
    func_passes = join(["canonicalize", "cse", "canonicalize", transform_passes], ",")
    if lower_comms
        func_passes =
            func_passes *
            ",enzyme-hlo-generate-td{patterns=" *
            lower_passes_str *
            "},transform-interpreter,enzyme-hlo-remove-transform"
    end
    if CONCATS_TO_DUS[]
        func_passes *= ",enzyme-hlo-generate-td{patterns=concat_to_onedim_dus},transform-interpreter,enzyme-hlo-remove-transform"
    end
    passes = String[]
    if compile_options.inline
        push!(passes, "inline{default-pipeline=canonicalize max-iterations=4}")
    end
    if sroa
        push!(passes, "propagate-constant-bounds")
        if DUMP_LLVMIR[]
            push!(
                passes,
                "sroa-wrappers{dump_prellvm=true dump_postllvm=true instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
            )
        else
            push!(
                passes,
                "sroa-wrappers{instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
            )
        end
        push!(passes, "canonicalize")
        push!(
            passes,
            "sroa-wrappers{instcombine=false instsimplify=true $(SROA_ATTRIBUTOR[] ? "" : "attributor=false")}",
        )
        push!(passes, "libdevice-funcs-raise")
        push!(passes, "canonicalize")
        push!(passes, "remove-duplicate-func-def")
    end
    push!(passes, func_passes)
    return join(passes, ',')
end

# TODO(#2251) we want to be able to run the more advanced passes via transform dialect as an enzyme intermediate
# However, this errs as we cannot attach the transform with to the funcop itself [as we run a functionpass].
const enzyme_pass::String = "enzyme{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true}\"}"

function impulse_pass(;
    debug_dump::Bool=DEBUG_PROBPROG_DUMP_VALUE[],
    disable_optimizations::Bool=DEBUG_PROBPROG_DISABLE_OPT[],
)
    if !disable_optimizations
        # TODO(#2063): Add impulse optimization passes
    end
    return "expand-impulse{debug-dump=$debug_dump postpasses=\"arith-raise{stablehlo=true}\"}"
end

function run_pass_pipeline!(mod, pass_pipeline, key=""; enable_verifier=true)
    pm = MLIR.IR.PassManager()
    MLIR.IR.enable_verifier!(pm, enable_verifier)
    opm = MLIR.IR.OpPassManager(pm)
    MLIR.IR.add_pipeline!(opm, pass_pipeline)
    MLIR.IR.run!(pm, MLIR.IR.Operation(mod), key)
    return mod
end

function run_pass_pipeline!(
    mod, propagation_options::ShardyPropagationOptions; enable_verifier=true
)
    pm = MLIR.IR.PassManager()
    MLIR.IR.enable_verifier!(pm, enable_verifier)
    opm = MLIR.IR.OpPassManager(pm)
    # TODO: why isn't this being auto-generated?
    @ccall MLIR.API.mlir_c.addSdyPropagationPipeline(
        opm::MLIR.API.MlirOpPassManager,
        propagation_options.keep_sharding_rules::UInt8,
        propagation_options.conservative_propagation::UInt8,
        propagation_options.debug_sharding_origins::UInt8,
        propagation_options.debug_propagation_edge_sharding::UInt8,
        propagation_options.skip_convert_to_reshard::UInt8,
        propagation_options.skip_inline::UInt8,
        propagation_options.enable_insert_explicit_collectives::UInt8,
    )::Cvoid
    MLIR.IR.run!(pm, mod, "sdy_prop")
    return mod
end

function create_pass_failure_zip(f, args)
    try
        # Create a temporary directory for the files
        temp_dir = mktempdir(; prefix="reactant_failure_", cleanup=false)

        function_name = string(f)
        function_name = replace(function_name, "!" => "_")
        Reactant.Serialization.export_to_reactant_script(
            f,
            args...;
            function_name,
            output_dir=temp_dir,
            compile_options=Reactant.CompileOptions(; optimization_passes=:none),
            try_zip_on_failure=false,
        )

        # Create the zip file
        zip_path = temp_dir * ".zip"
        temp_files = readdir(temp_dir; join=true)
        run(pipeline(`$(p7zip()) a -tzip $zip_path $temp_files`, devnull))

        return zip_path
    catch e
        @error "Failed to create debug zip file" exception = e
        return nothing
    end
end

# helper for debug purposes: String -> Text
function run_pass_pipeline_on_source(source, pass_pipeline; enable_verifier=true)
    return MLIR.IR.with_context() do _
        mod = parse(MLIR.IR.Module, source)
        run_pass_pipeline!(mod, pass_pipeline; enable_verifier)
        MLIR.IR.verifyall(MLIR.IR.Operation(mod); debug=true)
        Text(repr(mod))
    end
end

function __get_compile_options_and_kwargs(;
    compile_options::Union{Missing,CompileOptions}=missing,
    optimize::Union{Bool,Symbol,String}=true,
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
    shardy_passes::Union{Symbol,ShardyPropagationOptions}=:post_sdy_propagation,
    optimize_then_pad::Bool=true,
    optimize_communications::Union{Bool,OptimizeCommunicationOptions}=true,
    assert_nonallocating::Bool=false,
    donated_args::Symbol=:auto,
    sync::Bool=false,
    xla_debug_options=(;),
    xla_executable_build_options=(;),
    xla_compile_options=(;),
    strip=:all,
    strip_llvm_debuginfo=false,
    kwargs...,
)
    return (
        Reactant.__compile_options_from_kwargs(;
            compile_options,
            optimize,
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
            xla_debug_options,
            xla_executable_build_options,
            xla_compile_options,
            strip,
        ),
        kwargs,
    )
end

function get_optimize_comms_passes(options::Bool)
    if !options
        return [
            "enzyme-hlo-generate-td{patterns=lower_rotate;lower_wrap;lower_extend;lower_updatewithoutcorners}",
            "transform-interpreter",
            "enzyme-hlo-remove-transform",
        ]
    end
    return get_optimize_comms_passes(OptimizeCommunicationOptions())
end

function get_optimize_comms_passes(options::OptimizeCommunicationOptions)
    options_str = String(options)
    res = [
        "enzyme-hlo-generate-td{patterns=concat_to_onedim_dus;concat_to_onedim_dusslice;concatreshape_to_onedim_dus}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        "enzyme-hlo-generate-td{patterns=reshape_to_broadcast;recognize_multirotate;use_multirotate_neutral_result;recognize_multislice}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        options_str,
        "enzyme-hlo-generate-td{patterns=lower_rotate;lower_wrap;lower_extend;lower_updatewithoutcorners;lower_multislice}",
        "transform-interpreter",
        "enzyme-hlo-remove-transform",
        options_str,
    ]
    return res
end

function get_stablehlo_to_hlo_passes(; stablehlo_to_mhlo::Bool=true)
    passes = (
        "func.func(stablehlo-ext-chlo-recompose-ops)",
        "symbol-dce",
        "func.func(chlo-legalize-to-high-level-mhlo)",
        "func.func(chlo-legalize-to-stablehlo)",
    )
    if stablehlo_to_mhlo
        passes = (passes..., "stablehlo-legalize-to-hlo")
    end
    passes = (
        passes..., "canonicalize", "func.func(stablehlo-ext-sink-constants-to-control-flow)"
    )
    return passes
end
