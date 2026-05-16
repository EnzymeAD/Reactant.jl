module Compiler

using Reactant_jll
using Libdl: dlsym
using LinearAlgebra: BlasInt
import p7zip_jll: p7zip

import ..Reactant:
    Reactant,
    MLIR,
    XLA,
    Sharding,
    TracedRArray,
    TracedRNumber,
    RArray,
    OrderedIdDict,
    make_tracer,
    TracedToConcrete,
    append_path,
    TracedType
import Reactant: OptimizeCommunicationOptions, ShardyPropagationOptions, CompileOptions
using Reactant_jll: Reactant_jll

include("Macros.jl")
include("OptimizationPasses.jl")
include("Codegen.jl")
include("Thunk.jl")

const DEBUG_PRINT_CODEGEN = Ref(false)

const __module_gc_vector = Dict{MLIR.IR.Module,Vector{Union{TracedRArray,TracedRNumber}}}()

function guard_from_gc_for_module(mod::MLIR.IR.Module, x)
    if !haskey(__module_gc_vector, mod)
        __module_gc_vector[mod] = Union{TracedRArray,TracedRNumber}[x]
    else
        push!(__module_gc_vector[mod], x)
    end
    @assert __module_gc_vector[mod][end] === x
    return nothing
end

function release_guard_from_gc_for_module(mod::MLIR.IR.Module)
    delete!(__module_gc_vector, mod)
    return nothing
end

function compile_mlir(
    ctx,
    f,
    args;
    client=nothing,
    drop_unsupported_attributes=false,
    try_zip_on_failure::Bool=true,
    kwargs...,
)
    client = client !== nothing ? client : XLA.default_backend()
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

        # Wrap compile_mlir! to catch pass pipeline failures
        mlir_fn_res = try
            compile_mlir!(
                mod,
                f,
                args,
                compile_options;
                backend,
                runtime=XLA.runtime(client),
                client,
                kwargs_inner...,
            )
        catch e
            # Check if this is a pass pipeline failure
            error_msg = string(e)
            if contains(error_msg, "failed to run pass manager")
                # Prevent infinite recursion
                if !try_zip_on_failure
                    rethrow(
                        ErrorException(
                            "Compilation failed and we were unable to create a debug zip \
                            file.\nPlease report this issue at: \
                            https://github.com/EnzymeAD/Reactant.jl/issues\n\
                            Original error: $(error_msg)",
                        ),
                    )
                end

                # Create a debug zip file with the unoptimized IR
                zip_path = create_pass_failure_zip(f, args)
                if zip_path !== nothing
                    rethrow(
                        ErrorException(
                            "Compilation failed during pass pipeline execution.\n\
                            A debug zip file has been created at: $(zip_path)\n\
                            Please upload this file when reporting the issue at: \
                            https://github.com/EnzymeAD/Reactant.jl/issues\n\
                            Original error: $(error_msg)"
                        ),
                    )
                end
            end
            rethrow()
        end

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        if drop_unsupported_attributes
            # Drop some of our attributes
            run_pass_pipeline!(
                mod, "drop-unsupported-attributes", "drop_enzymexla_attributes"
            )
        end

        return mod, mlir_fn_res
    finally
        MLIR.IR.deactivate(ctx)
    end

    return results
end

const PartitionKA = Ref{Bool}(true)

const cuindexBitWidth = Ref{Int}(32)
const cubinFormat = Ref{String}("bin")
const cuOptLevel = Ref{Int}(2)

# Whatever the relevant highest version from our LLVM is within NVPTX.td
# Or more specifically looking at clang/lib/Driver/ToolChains/Cuda.cpp:682
#  We see relevant ptx version is CUDA 13.2 -> 92
#                                      13.0 -> 90
#                                      12.9 -> 88
#                                      12.6 -> 85
#                                      12.2 -> 82
#                                      11.8 -> 78
function cubinFeatures()
    fallback = "+ptx80"
    ver = MLIR.API.ReactantCudaDriverGetVersion()
    # No cuda available
    if ver == 0
        return fallback
    end
    ver2 = MLIR.API.ReactantHermeticCudaGetVersion()
    ver = min(ver, ver2)
    major, ver = divrem(ver, 1000)
    minor, _ = divrem(ver, 10)
    cuver_map = Dict([
        # For CUDA 13+, you have to go through the documentation of each minor version
        # (https://developer.nvidia.com/cuda-toolkit-archive) and look at the PTX ISA page.
        (132, 92),
        (131, 91),
        (130, 90),
        # From https://github.com/llvm/llvm-project/blob/7d7cd745af221c8690ea6deb2dfbf232658158cd/clang/lib/Driver/ToolChains/Cuda.cpp#L682
        (129, 88),
        (128, 87),
        (126, 85),
        (125, 85),
        (124, 84),
        (123, 83),
        (122, 82),
        (121, 81),
        (120, 80),
        (118, 78),
        (117, 77),
        (116, 76),
        (115, 75),
        (114, 74),
        (113, 73),
        (112, 72),
        (111, 71),
        (110, 70),
        (102, 65),
        (101, 64),
        (100, 63),
        (92, 61),
        (91, 61),
        (90, 60),
    ])
    mver = major * 10 + minor
    if !in(mver, keys(cuver_map))
        return fallback
    end
    ptx = cuver_map[mver]
    return "+ptx$ptx"
end

function activate_raising!(is_raising::Bool)
    stack = get!(task_local_storage(), :reactant_is_raising) do
        Bool[]
    end::Vector{Bool}
    push!(stack, is_raising)
    return nothing
end

function deactivate_raising!(is_raising::Bool)
    key = :reactant_is_raising
    is_raising === last(task_local_storage(key)::Vector{Bool}) ||
        error("Deactivating wrong Reactant raising context")
    return pop!(task_local_storage(key)::Vector{Bool})
end

function raising(; throw_error::Bool=true)
    key = :reactant_is_raising
    if !(
        haskey(task_local_storage(), key) &&
        !Base.isempty(task_local_storage(key)::Vector{Bool})
    )
        throw_error && error("No Reactant raising context")
    end
    return last(task_local_storage(key)::Vector{Bool})
end

function raising!(f, is_raising::Bool)
    activate_raising!(is_raising)
    try
        return f()
    finally
        deactivate_raising!(is_raising)
    end
end

function compile_mlir!(
    mod,
    f,
    args,
    compile_options::CompileOptions,
    debugcache=default_debugcache(),
    callcache=default_callcache(),
    sdycache=default_sdycache(),
    sdygroupidcache=default_sdygroupidcache();
    fn_kwargs=(),
    backend="gpu",
    runtime::Union{Val{:PJRT},Val{:IFRT}},
    legalize_stablehlo_to_mhlo::Bool=false,
    client=nothing,
    kwargs...,
)
    @assert MLIR.IR.current_context() == MLIR.IR.context(mod)
    client = client !== nothing ? client : XLA.default_backend()

    # Explicitly don't use with_block to avoid creating a closure, which creates
    # both compile-time and relocatability issues

    MLIR.IR.activate(mod)
    MLIR.IR.activate(MLIR.IR.body(mod))
    activate_callcache!(callcache)
    activate_debugcache!(debugcache)
    activate_sdycache!(sdycache)
    activate_sdygroupidcache!(sdygroupidcache)

    # Save in the TLS whether we are raising.  We identify that condition by
    # checking whether the user set an explicit list of passes, or chose
    # `raise=true` to use the default passes.
    raise = compile_options.raise
    if (backend == "tpu" || backend == "neuron" || backend == "trainium") && raise isa Bool
        raise = true
    end
    is_raising = raise isa String || raise
    activate_raising!(is_raising)

    fnname = string(f)
    mlir_fn_res = try
        Reactant.TracedUtils.make_mlir_fn(
            f,
            args,
            fn_kwargs,
            fnname,
            true;
            runtime,
            compile_options.optimize_then_pad,
            kwargs...,
        )
    finally
        deactivate_raising!(is_raising)
        deactivate_sdycache!(sdycache)
        deactivate_sdygroupidcache!(sdygroupidcache)
        deactivate_callcache!(callcache)
        deactivate_debugcache!(debugcache)
        MLIR.IR.deactivate(MLIR.IR.body(mod))
        clear_llvm_compiler_cache!(mod)
        release_guard_from_gc_for_module(mod)
        MLIR.IR.deactivate(mod)
    end
    (;
        fnwrapped,
        traced_result,
        seen_args,
        linear_args,
        skipped_args,
        in_tys,
        linear_results,
        skipped_results,
        is_sharded,
    ) = mlir_fn_res
    compiled_f = mlir_fn_res.f

    # Custom Kernels without Raising will lead to suboptimal codegen for is_sharded, force
    # raising
    if is_sharded
        is_raising = true
        raise isa Bool && (raise = true)
    end

    toolkit = XLA.CUDA_DATA_DIR[]

    if backend == "cpu" || backend == "tpu" || backend == "neuron" || backend == "trainium"
        kern = "lower-kernel{backend=cpu},canonicalize"
        if backend == "tpu" || backend == "neuron" || backend == "trainium"
            jit = "lower-jit{openmp=$(OpenMP[]) backend=cpu},symbol-dce,strip-debuginfo"
        else
            jit = "lower-jit{openmp=$(OpenMP[]) backend=cpu},symbol-dce"
        end
    else
        kern = if is_raising
            "lower-kernel{backend=cpu},symbol-dce,canonicalize"
        else
            "lower-kernel,canonicalize"
        end

        device_properties = XLA.device_properties(XLA.default_device(client))
        cubinChip = "sm_$(device_properties.major)$(device_properties.minor)"

        if DEBUG_KERNEL[]
            curesulthandler = dlsym(
                Reactant_jll.libReactantExtra_handle, "ReactantHandleCuResult"
            )
            @assert curesulthandler !== nothing
            curesulthandler = Base.reinterpret(UInt, curesulthandler)
            extra_lowerjit_options = "debug=true cuResultHandlerPtr=$curesulthandler "
        else
            extra_lowerjit_options = ""
        end
        jit = "lower-jit{$(extra_lowerjit_options)cuOptLevel=$(cuOptLevel[]) cubinFormat=$(cubinFormat[]) indexBitWidth=$(cuindexBitWidth[])  cubinChip=$(cubinChip) cubinFeatures=$(cubinFeatures()) run_init=true toolkitPath=$toolkit},symbol-dce"
    end

    recognize_comms = true
    lower_comms = true
    if is_sharded && (
        compile_options.shardy_passes == :to_mhlo_shardings ||
        compile_options.shardy_passes == :post_sdy_propagation ||
        compile_options.shardy_passes isa ShardyPropagationOptions
    )
        lower_comms = false
    end

    opt_passes = optimization_passes(
        compile_options; sroa=true, recognize_comms, lower_comms, backend, is_sharded
    )
    opt_passes2 = optimization_passes(
        compile_options; sroa=false, recognize_comms, lower_comms, backend, is_sharded
    )

    raise_passes = if raise isa String
        # Raising passes were specified
        raise
    elseif raise

        # Raise enabled but use default passes
        # TODO(#2240) remove redundant libdevice raise after fixing phase ordering
        result =
            "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,func.func(kernelcast),raise-affine-to-stablehlo{strip_llvm_debuginfo=$(compile_options.strip_llvm_debuginfo) prefer_while_raising=false dump_failed_lockstep=$(DUMP_FAILED_LOCKSTEP[])},canonicalize,arith-raise{stablehlo=true}," *
            opt_passes2

        if DUS_TO_CONCAT[]
            opt_passes3 = optimization_passes(
                compile_options;
                sroa=false,
                dus_to_concat=true,
                recognize_comms,
                lower_comms,
                backend,
                is_sharded,
            )
            result = result * "," * opt_passes3
        end
        result
    else
        "canonicalize"
    end

    blas_int_width = sizeof(BlasInt) * 8
    lower_enzymexla_linalg_pass = join(
        [
            "lower-enzymexla-linalg{backend=$backend blas_int_width=$blas_int_width}",
            "lower-enzymexla-blas{backend=$backend blas_int_width=$blas_int_width}",
            "lower-enzymexla-lapack{backend=$backend blas_int_width=$blas_int_width}",
        ],
        ",",
    )
    lower_enzymexla_ml_pass = "lower-enzymexla-ml"
    lower_enzymexla_mpi_pass = "lower-enzymexla-mpi{backend=$backend}"
    lower_enzymexla_passes = join(
        [lower_enzymexla_linalg_pass, lower_enzymexla_ml_pass, lower_enzymexla_mpi_pass],
        ",",
    )

    legalize_chlo_to_stablehlo =
        if legalize_stablehlo_to_mhlo || compile_options.legalize_chlo_to_stablehlo
            get_stablehlo_to_hlo_passes(; stablehlo_to_mhlo=legalize_stablehlo_to_mhlo)
        else
            ()
        end

    legal_to_run_shardy_passes = compile_options.optimization_passes === :all

    # Raise any triton kernel that might exist as a custom call
    # We will lower them back later on, but having the full triton IR enables
    # optimizations / differentiation, so we unconditionally do it
    if compile_options.raise_triton_custom_call
        run_pass_pipeline!(mod, "raise-triton-custom-call", "raise_triton_custom_call")
    end

    if compile_options.optimization_passes === :all
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        lower_enzymexla_passes,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                        lower_enzymexla_passes,
                        jit,
                    ]
                end,
                ",",
            ),
            "all",
        )
    elseif compile_options.optimization_passes === :before_kernel
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    ["mark-func-memory-effects", opt_passes]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                    ]
                end,
                ',',
            ),
            "before_kernel",
        )
    elseif compile_options.optimization_passes === :before_jit
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                    ]
                end,
                ',',
            ),
            "before_jit",
        )
    elseif compile_options.optimization_passes === :before_raise
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    ["mark-func-memory-effects", opt_passes]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                    ]
                end,
                ',',
            ),
            "before_raise",
        )
    elseif compile_options.optimization_passes === :no_enzyme
        run_pass_pipeline!(
            mod,
            join(
                [
                    "mark-func-memory-effects",
                    opt_passes,
                    "enzyme-batch",
                    opt_passes2,
                    enzyme_pass,
                    opt_passes2,
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    legalize_chlo_to_stablehlo...,
                    opt_passes2,
                ],
                ',',
            ),
            "no_enzyme",
        )
    elseif compile_options.optimization_passes === :probprog
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        impulse_pass(),
                        "lower-impulse-to-stablehlo{backend=$backend}",
                        "outline-enzyme-regions",
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        (
                            if compile_options.legalize_chlo_to_stablehlo
                                ["func.func(chlo-legalize-to-stablehlo)"]
                            else
                                []
                            end
                        )...,
                        opt_passes2,
                        lower_enzymexla_passes,
                        "lower-impulse-trace-ops{backend=$backend}",
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        impulse_pass(),
                        "lower-impulse-to-stablehlo{backend=$backend}",
                        "outline-enzyme-regions",
                        enzyme_pass,
                        opt_passes2,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        (
                            if compile_options.legalize_chlo_to_stablehlo
                                ["func.func(chlo-legalize-to-stablehlo)"]
                            else
                                []
                            end
                        )...,
                        opt_passes2,
                        kern,
                        raise_passes,
                        lower_enzymexla_passes,
                        "lower-impulse-trace-ops{backend=$backend}",
                        jit,
                    ]
                end,
                ",",
            ),
            "impulse",
        )
    elseif compile_options.optimization_passes === :only_enzyme
        run_pass_pipeline!(
            mod,
            join(
                [
                    "mark-func-memory-effects",
                    "enzyme-batch",
                    enzyme_pass,
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                ],
                ',',
            ),
            "only_enzyme",
        )
    elseif compile_options.optimization_passes === :noopt
        run_pass_pipeline!(
            mod,
            join(
                [
                    "mark-func-memory-effects",
                    "enzyme-batch",
                    enzyme_pass,
                    "canonicalize",
                    "remove-unnecessary-enzyme-ops",
                    "enzyme-simplify-math",
                    lower_enzymexla_passes,
                    jit,
                ],
                ',',
            ),
            "only_enzyme",
        )
    elseif compile_options.optimization_passes === :after_enzyme
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        enzyme_pass,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        lower_enzymexla_passes,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        "enzyme-batch",
                        enzyme_pass,
                        "canonicalize",
                        "remove-unnecessary-enzyme-ops",
                        "enzyme-simplify-math",
                        legalize_chlo_to_stablehlo...,
                        opt_passes2,
                        kern,
                        raise_passes,
                        lower_enzymexla_passes,
                        jit,
                    ]
                end,
                ',',
            ),
            "after_enzyme",
        )
    elseif compile_options.optimization_passes === :before_enzyme
        run_pass_pipeline!(
            mod,
            join(
                if compile_options.raise_first
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        kern,
                        raise_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math",
                        lower_enzymexla_passes,
                        jit,
                    ]
                else
                    [
                        "mark-func-memory-effects",
                        opt_passes,
                        "enzyme-batch",
                        opt_passes2,
                        enzyme_pass,
                        "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math",
                        kern,
                        raise_passes,
                        lower_enzymexla_passes,
                        jit,
                    ]
                end,
                ',',
            ),
            "before_enzyme",
        )
    elseif compile_options.optimization_passes === :canonicalize
        run_pass_pipeline!(mod, "mark-func-memory-effects,canonicalize", "canonicalize")
    elseif compile_options.optimization_passes === :just_batch
        run_pass_pipeline!(mod, "enzyme-batch", "enzyme-batch")
    elseif compile_options.optimization_passes isa String
        run_pass_pipeline!(mod, compile_options.optimization_passes, "custom_pass")
    end

    if compile_options.optimization_passes isa Symbol &&
        compile_options.optimization_passes === :all &&
        (
            compile_options.transpose_propagate === :up ||
            compile_options.reshape_propagate === :up
        )
        # We tried propagating reshapes and transposes up. If at this point we are left
        # with them, we propagate them down to minimize the number of Ops in the IR.
        # Since this might enable certain raising, we do push down -> push up -> push down
        common_kwargs = (;
            recognize_comms,
            lower_comms,
            backend,
            is_sharded,
            raise_shlo_to_blas_lapack=false,
        )
        opt_passes_down = optimization_passes(
            Reactant.__compile_options_with_reversed_propagation(compile_options);
            common_kwargs...,
        )
        opt_passes_up = optimization_passes(compile_options; common_kwargs...)
        run_pass_pipeline!(
            mod,
            join([opt_passes_down, opt_passes_up, opt_passes_down], ","),
            "post_op_transpose_reshape",
        )
    end

    if backend == "cuda" && compile_options.cudnn_hlo_optimize
        run_pass_pipeline!(mod, "enzymexla-cudnn-hlo-opt", "cudnn-hlo-opt")
    end

    if compile_options.lower_triton
        run_pass_pipeline!(mod, "lower-triton", "lower_triton")
    end

    # Now we resolve paddings if `optimize_then_pad`
    if compile_options.optimize_then_pad
        padded_inputs = IdDict()
        has_padded_inputs = false
        for (k, v) in seen_args
            v isa Reactant.TracedType || continue
            if Reactant.has_padding(k)
                has_padded_inputs = true
                padded_inputs[v] = Reactant.get_padding(k)
            end
        end

        if has_padded_inputs
            MLIR.IR.DUMP_MLIR_ALWAYS[] && MLIR.IR.dump_mlir(mod, nothing, "pre_padding")

            in_tys_padded = Vector{MLIR.IR.Type}(undef, length(linear_args))
            input_arg_padded_idxs = Int[]
            for (i, arg) in enumerate(linear_args)
                if haskey(padded_inputs, arg)
                    push!(input_arg_padded_idxs, i)
                    in_tys_padded[i] = MLIR.IR.TensorType(
                        collect(Int, reverse(size(arg) .+ padded_inputs[arg])),
                        MLIR.IR.Type(Reactant.unwrapped_eltype(arg)),
                    )
                else
                    in_tys_padded[i] = in_tys[i]
                end
            end

            out_tys_padded = Vector{MLIR.IR.Type}(undef, length(linear_results))
            output_res_padded_idxs = Int[]
            for (i, res) in enumerate(linear_results)
                if haskey(padded_inputs, res)
                    push!(output_res_padded_idxs, i)
                    out_tys_padded[i] = MLIR.IR.TensorType(
                        collect(Int, reverse(size(res) .+ padded_inputs[res])),
                        MLIR.IR.Type(Reactant.unwrapped_eltype(res)),
                    )
                else
                    out_tys_padded[i] = Reactant.TracedUtils.transpose_ty(
                        Reactant.Ops.mlir_type(res)
                    )
                end
            end

            fnname_old = fnname
            fnname = string(f, "_padded")
            func_with_padding = MLIR.Dialects.func.func_(;
                sym_name=fnname,
                function_type=MLIR.IR.FunctionType(in_tys_padded, out_tys_padded),
                arg_attrs=MLIR.IR.getattr(compiled_f, "arg_attrs"),
                res_attrs=MLIR.IR.getattr(compiled_f, "res_attrs"),
                no_inline=MLIR.IR.getattr(compiled_f, "no_inline"),
                body=MLIR.IR.Region(),
                sym_visibility=MLIR.IR.getattr(compiled_f, "private"),
            )
            fnbody = MLIR.IR.Block(
                in_tys_padded,
                [
                    MLIR.IR.Location(
                        MLIR.API.mlirValueGetLocation(
                            MLIR.IR.argument(
                                MLIR.IR.first_block(MLIR.IR.region(compiled_f, 1)), i
                            ),
                        ),
                    ) for i in 1:length(linear_args)
                ],
            )
            push!(MLIR.IR.region(func_with_padding, 1), fnbody)
            MLIR.IR.activate(fnbody)
            push!(MLIR.IR.body(mod), func_with_padding)

            try
                call_args = MLIR.IR.Value[
                    MLIR.IR.argument(fnbody, i) for i in 1:length(linear_args)
                ]

                for i in input_arg_padded_idxs
                    arg = linear_args[i]
                    padding = padded_inputs[arg]

                    block_arg = MLIR.IR.argument(fnbody, i)
                    unpad_op = Reactant.TracedUtils.unpad_val_op(
                        block_arg, reverse(padding), reverse(size(arg) .+ padding)
                    )

                    call_args[i] = MLIR.IR.result(unpad_op, 1)
                end

                ftype = MLIR.IR.Type(MLIR.IR.getattr(compiled_f, "function_type"))
                call_op = MLIR.Dialects.func.call(
                    call_args;
                    result_0=[MLIR.IR.result(ftype, i) for i in 1:MLIR.IR.nresults(ftype)],
                    callee=MLIR.IR.FlatSymbolRefAttribute(fnname_old),
                )

                results = MLIR.IR.Value[
                    MLIR.IR.result(call_op, i) for i in 1:MLIR.IR.nresults(call_op)
                ]

                for i in output_res_padded_idxs
                    res = linear_results[i]
                    padding = padded_inputs[res]

                    pad_op = MLIR.Dialects.stablehlo.pad(
                        results[i],
                        Reactant.TracedUtils.promote_to(
                            TracedRNumber{Reactant.unwrapped_eltype(res)}, 0
                        ).mlir_data;
                        edge_padding_low=MLIR.IR.DenseArrayAttribute(
                            fill(0, length(padding))
                        ),
                        edge_padding_high=MLIR.IR.DenseArrayAttribute(
                            collect(reverse(padding))
                        ),
                        interior_padding=MLIR.IR.DenseArrayAttribute(
                            fill(0, length(padding))
                        ),
                    )

                    results[i] = MLIR.IR.result(pad_op, 1)
                end

                MLIR.Dialects.func.return_(results)
            finally
                MLIR.IR.deactivate(fnbody)
            end

            # we just need the ops to potentially remove slices / paddings
            if compile_options.optimization_passes === :all
                run_pass_pipeline!(
                    mod,
                    join(
                        [
                            opt_passes,
                            "canonicalize",
                            "cse",
                            "canonicalize",
                            opt_passes2,
                            lower_enzymexla_passes,
                            jit,
                        ],
                        ",",
                    ),
                    "mid_pad_opts",
                )
            end

            MLIR.IR.setattr!(compiled_f, "sym_visibility", MLIR.IR.Attribute("private"))
            run_pass_pipeline!(
                mod,
                "inline{default-pipeline=canonicalize max-iterations=4}",
                "inline_pad_opts",
            )

            compiled_f = func_with_padding
            in_tys = in_tys_padded
        end
    end

    if compile_options.multifloat !== nothing
        run_pass_pipeline!(mod, String(compile_options.multifloat), "multifloat")
    end

    # shardy passes
    use_shardy_partitioner = false
    result_shardings = missing
    if is_sharded && legal_to_run_shardy_passes
        mod_copied = copy(mod)

        if compile_options.shardy_passes isa ShardyPropagationOptions
            run_pass_pipeline!(mod_copied, compile_options.shardy_passes)
            run_pass_pipeline!(mod_copied, "sdy-close-shardings", "sdy_close_shardings")
        else
            run_pass_pipeline!(
                mod_copied,
                join(["sdy-propagation-pipeline", "sdy-close-shardings"], ","),
                "sdy_prop_capture_res_shardings",
            )
        end

        func_op_new_module = MLIR.IR.@dispose sym_table = MLIR.IR.SymbolTable(mod_copied) begin
            MLIR.IR.lookup(sym_table, fnname)
        end

        result_attrs = MLIR.IR.getattr(func_op_new_module, "res_attrs")
        if result_attrs !== nothing
            result_shardings = Vector{Union{Sharding.NamedSharding,Sharding.Replicated}}(
                undef, length(result_attrs)
            )
            for i in 1:length(result_attrs)
                result_shardings[i] = Sharding.sdy_sharding_to_reactant_sharding(
                    result_attrs[i - 1], mlir_fn_res.global_device_ids, mod_copied
                )
            end
        else
            result_shardings = [Sharding.Replicated() for _ in 1:length(linear_results)]
        end

        if compile_options.shardy_passes === :none
            use_shardy_partitioner = true
        elseif compile_options.shardy_passes === :post_sdy_propagation
            use_shardy_partitioner = true
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-propagation-pipeline",
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "func.func(sdy-reshard-to-collectives)",
                    ],
                    ",",
                ),
                "post_sdy_propagation",
            )
        elseif compile_options.shardy_passes isa ShardyPropagationOptions
            run_pass_pipeline!(mod, compile_options.shardy_passes)
            # sdy passes are run deep inside the XLA compiler. So the only way to respect
            # the options is to export them to MHLO shardings
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "xla-sdy-stablehlo-export-pipeline",
                    ],
                    ",",
                ),
                "sdy_export",
            )
        elseif compile_options.shardy_passes === :to_mhlo_shardings
            run_pass_pipeline!(
                mod,
                join(
                    [
                        "sdy-propagation-pipeline",
                        "sdy-close-shardings",
                        get_optimize_comms_passes(
                            compile_options.optimize_communications
                        )...,
                        "func.func(sdy-reshard-to-collectives)",
                        "xla-sdy-stablehlo-export-pipeline",
                    ],
                    ",",
                ),
                "to_mhlo_shardings",
            )
        end
    end

    if compile_options.optimization_passes !== :none
        run_pass_pipeline!(mod, "mark-func-memory-effects", "mark-func-memory-effects")
    end

    if compile_options.strip === :all
        run_pass_pipeline!(mod, "strip-debuginfo", "strip-debuginfo")
    elseif compile_options.strip isa Vector && length(compile_options.strip) != 0
        run_pass_pipeline!(
            mod,
            "trim-callsites{to_trim=$(join(compile_options.strip, ";"))}",
            "trim-callsites",
        )
    else
        @assert compile_options.strip === :none
    end

    func_op = MLIR.IR.@dispose sym_table = MLIR.IR.SymbolTable(mod) begin
        MLIR.IR.lookup(sym_table, fnname)
    end
    fnbody = MLIR.IR.first_block(MLIR.IR.region(func_op, 1))::MLIR.IR.Block
    ret = MLIR.IR.terminator(fnbody)::MLIR.IR.Operation

    preserved_args = Tuple{TracedType,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedType[]
    results_mask = falses(length(results))

    for (i, op) in enumerate(results)
        if !MLIR.IR.is_block_arg(op) ||
            !Reactant.TracedUtils.has_idx(linear_results[i], :args) # new buffer
            push!(nresults, op)
            push!(linear_results2, linear_results[i])
            results_mask[i] = true
            continue
        end

        push!(preserved_args, (linear_results[i], MLIR.IR.block_arg_num(op)))
    end

    MLIR.IR.dispose(ret)

    MLIR.IR.@with_block fnbody begin
        return MLIR.Dialects.func.return_(nresults)
    end

    out_tys2 = [MLIR.IR.type(a) for a in nresults]

    res_attrs = MLIR.IR.getattr(compiled_f, "res_attrs")
    if res_attrs isa MLIR.IR.Attribute
        res_attrs = MLIR.IR.Attribute[
            res_attrs[i - 1] for (i, present) in enumerate(results_mask) if present
        ]
    end

    if result_shardings !== missing
        result_shardings_after_masking = eltype(result_shardings)[]
        for (i, present) in enumerate(results_mask)
            if present
                push!(result_shardings_after_masking, result_shardings[i])
            end
        end
    else
        result_shardings_after_masking = missing
    end

    func3 = MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=MLIR.IR.FunctionType(in_tys, out_tys2),
        arg_attrs=MLIR.IR.getattr(compiled_f, "arg_attrs"),
        res_attrs,
        no_inline=MLIR.IR.getattr(compiled_f, "no_inline"),
        body=MLIR.IR.Region(),
    )
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func3, 1), MLIR.IR.region(compiled_f, 1))

    push!(MLIR.IR.body(mod), func3)

    mem = MLIR.IR.getattr(compiled_f, "enzymexla.memory_effects")
    if !(mem isa Nothing)
        MLIR.IR.setattr!(func3, "enzymexla.memory_effects", mem)
    end

    MLIR.IR.dispose(compiled_f)

    # Add a `donated` attr to the function arguments. This doesn't affect XLA, but lets us
    # check which arguments were donated.
    preserved_args_idx = last.(preserved_args)
    donated_args_mask = Vector{Bool}(undef, length(linear_args))
    for (i, arg) in enumerate(linear_args)
        if compile_options.donated_args == :auto
            if (i - 1) ∉ preserved_args_idx
                donated_args_mask[i] = true

                residx = findfirst(Base.Fix1(===, arg), linear_results2)
                if residx !== nothing && in_tys[i] == out_tys2[residx]
                    MLIR.API.mlirFuncSetArgAttr(
                        func3,
                        i - 1,
                        "tf.aliasing_output",
                        MLIR.IR.Attribute(Int32(residx - 1)),
                    )
                end
            else
                donated_args_mask[i] = false
            end
        else # :none
            donated_args_mask[i] = false
        end
    end

    # drop certain operations from the module if using TPU or Trainium backend
    if backend == "tpu" || backend == "neuron" || backend == "trainium"
        for op in collect(MLIR.IR.body(mod))
            if MLIR.IR.dialect(op) == :llvm
                MLIR.IR.dispose(op)
            end
        end
    end

    if compile_options.assert_nonallocating
        if length(linear_args) - length(preserved_args_idx) != length(nresults)
            str = sprint() do io
                Base.show(IOContext(io, :debug => true), func3)
            end
            throw(
                AssertionError(
                    """length(preserved_args_idx) = $(length(preserved_args_idx))
             donated = length(linear_args) - length(preserved_args_idx) = $(length(linear_args) - length(preserved_args_idx))
                    length(nresults) = $(length(nresults))
                    linear_args = $linear_args
                    linear_results = $linear_results
                    $(join((MLIR.IR.argument(fnbody, i) for i in 1:length(in_tys)), ", "))
                    preserved_args = $(preserved_args_idx)
                    $str
                    """,
                ),
            )
        end
    end

    concrete_result = make_tracer(
        OrderedIdDict(), traced_result, ("result",), TracedToConcrete; runtime
    )

    return Reactant.TracedUtils.CompiledMlirFnResult(
        fnwrapped,
        func3,
        traced_result,
        mlir_fn_res.result,
        seen_args,
        ret,
        linear_args,
        skipped_args,
        in_tys,
        linear_results2,
        skipped_results,
        mlir_fn_res.num_partitions,
        mlir_fn_res.num_replicas,
        mlir_fn_res.is_sharded,
        preserved_args,
        concrete_result,
        mlir_fn_res.unique_meshes,
        mlir_fn_res.mutated_args,
        use_shardy_partitioner,
        result_shardings_after_masking,
        mlir_fn_res.global_device_ids,
        donated_args_mask,
        Reactant.TracedUtils.is_pure(func3),
    )
end

function __add_mhlo_attributes_and_name!(mod::MLIR.IR.Module, f; kwargs...)
    fname = string(f)
    length(fname) > 10 && (fname = fname[1:7] * "...")
    __add_mhlo_attributes_and_name!(mod, fname; kwargs...)
    return nothing
end

function __add_mhlo_attributes_and_name!(
    mod::MLIR.IR.Module, fname::String; num_partitions=1, num_replicas=1
)
    moduleop = MLIR.IR.Operation(mod)
    module_name = Reactant.TracedUtils.__lookup_unique_name_in_module(
        mod, "reactant_" * fname
    )
    module_name = MLIR.IR.Attribute(module_name)
    MLIR.IR.setattr!(moduleop, "mhlo.num_partitions", MLIR.IR.Attribute(num_partitions))
    MLIR.IR.setattr!(moduleop, "mhlo.num_replicas", MLIR.IR.Attribute(num_replicas))
    MLIR.IR.setattr!(
        moduleop, String(MLIR.API.mlirSymbolTableGetSymbolAttributeName()), module_name
    )
    return nothing
end

function __resolve_device_and_client(client, seen_args, linear_args, is_sharded)
    if is_sharded
        client === nothing && (client = XLA.default_backend())
        return client, nothing
    end

    device = nothing
    if length(linear_args) > 0
        devices_list = []
        for (k, v) in seen_args
            !(v isa TracedRArray || v isa TracedRNumber) && continue
            buffer = k.data isa Tuple ? only(k.data) : k.data
            push!(devices_list, XLA.device(buffer))
        end
        if !isempty(devices_list)
            if !allequal(devices_list)
                msg = "Expected all arguments to be on the same device, got:\n"
                for (i, device) in enumerate(unique(devices_list))
                    msg *= "    Device $(i): $(string(device))\n"
                end
                throw(ArgumentError(msg))
            end
            @assert allequal(devices_list) "All arguments must be on the same device: $(devices_list)"
            device = first(devices_list)
        end
    end

    if client === nothing
        if device !== nothing
            client = XLA.client(device)
        else
            client = XLA.default_backend()
            device = XLA.default_device(client)
        end
    else
        if device !== nothing
            @assert client == XLA.client(device) "client ($(client)) and XLA.client(device) ($(XLA.client(device))) must be the same"
        else
            device = XLA.default_device(client)
        end
    end

    return (client, device)
end

function compile_xla(f, args; kwargs...)
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        compile_xla(ctx, f, args; kwargs...)
    end
end

function compile_xla(
    ctx,
    f,
    args;
    before_xla_optimizations::Bool=false,
    client=nothing,
    serializable::Bool=false,
    kwargs...,
)
    client = client !== nothing ? client : XLA.default_backend()
    backend = XLA.platform_name(client)

    if backend == "CUDA"
        backend = "GPU"
    elseif backend == "CPU"
        backend = "cpu"
    end

    MLIR.IR.activate(ctx)
    try
        # compile function to MLIR module
        mod = MLIR.IR.Module(MLIR.IR.Location())

        compile_options, kwargs = __get_compile_options_and_kwargs(; kwargs...)

        mlir_fn_res = compile_mlir!(
            mod,
            f,
            args,
            compile_options;
            backend,
            runtime=XLA.runtime(client),
            client,
            kwargs...,
        )

        # Resolve client and device
        client, device = __resolve_device_and_client(
            client, mlir_fn_res.seen_args, mlir_fn_res.linear_args, mlir_fn_res.is_sharded
        )

        # Attach a name, and partitioning attributes to the module
        __add_mhlo_attributes_and_name!(
            mod, f; mlir_fn_res.num_partitions, mlir_fn_res.num_replicas
        )

        # Drop some of our attributes
        run_pass_pipeline!(mod, "drop-unsupported-attributes", "drop_enzymexla_attributes")

        # compile MLIR module to XLA executable
        global_device_ids = collect(Int64, mlir_fn_res.global_device_ids)
        mlir_fn_res.is_sharded && (device = nothing)

        # XLA.compile mutates the module, for serialization we need to keep a copy
        if serializable
            iobuffer = IOBuffer()
            show(IOContext(iobuffer, :debug => true), mod)
            module_string = String(take!(iobuffer))
        else
            module_string = ""
        end

        if before_xla_optimizations
            exec = nothing
            hlo_modules = XLA.HloModule(mod)
        else
            xla_compile_options = XLA.make_compile_options(;
                device_id=if mlir_fn_res.is_sharded
                    -1
                else
                    Int64(XLA.device_ordinal(device))
                end,
                xla_compile_options=compile_options.xla_compile_options,
                num_replicas=mlir_fn_res.num_replicas,
                num_partitions=mlir_fn_res.num_partitions,
                mesh_ids=mlir_fn_res.is_sharded ? global_device_ids : nothing,
                xla_debug_options=compile_options.xla_debug_options,
                xla_executable_build_options=merge(
                    (;
                        use_shardy_partitioner=mlir_fn_res.use_shardy_partitioner,
                        use_spmd_partitioning=mlir_fn_res.is_sharded,
                    ),
                    compile_options.xla_executable_build_options,
                ),
            )

            exec = XLA.compile(
                client,
                mod;
                compile_options=xla_compile_options,
                num_outputs=length(mlir_fn_res.linear_results),
                num_parameters=length(mlir_fn_res.linear_args),
                mlir_fn_res.is_sharded,
                mlir_fn_res.num_replicas,
                mlir_fn_res.num_partitions,
            )
            hlo_modules = XLA.get_hlo_modules(exec)
            hlo_modules = length(hlo_modules) == 1 ? only(hlo_modules) : hlo_modules
        end

        finalize(mod)

        return exec, hlo_modules, mlir_fn_res, device, client, module_string
    finally
        MLIR.IR.deactivate(ctx)
    end
end

function compile(f, args; kwargs...)
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        compile(ctx, f, args; kwargs...)
    end
end

function compile(ctx, f, args; kwargs...)
    compile_options, kwargs = __get_compile_options_and_kwargs(; kwargs...)

    exec, _, mlir_fn_res, device, client, str = compile_xla(
        ctx, f, args; compile_options, kwargs...
    )
    (; linear_args, seen_args, linear_results, preserved_args, concrete_result) =
        mlir_fn_res

    result_stores = Dict{Tuple,Symbol}()
    path_to_shard_info = mlir_fn_res.is_sharded ? Dict{Tuple,Symbol}() : nothing

    global_mesh_expr = if mlir_fn_res.unique_meshes === nothing
        :()
    elseif length(mlir_fn_res.unique_meshes) == 1
        only_mesh = only(mlir_fn_res.unique_meshes)
        :(Sharding.Mesh(
            thunk.global_device_ids, # same as only_mesh.global_device_ids
            0:(length(thunk.global_device_ids) - 1), # same as only_mesh.logical_device_ids
            $(only_mesh.axis_names),
            $(only_mesh.axis_sizes),
        ))
    else
        :(Sharding.Mesh(
            thunk.global_device_ids,
            0:(length(thunk.global_device_ids) - 1),
            (:flat_mesh,),
            (length(thunk.global_device_ids),),
        ))
    end

    ndevices = mlir_fn_res.is_sharded ? length(mlir_fn_res.global_device_ids) : 1

    # generate Julia `Thunk` code
    flatten_arg_names, flatten_code, resharded_inputs = codegen_flatten!(
        linear_args,
        seen_args,
        mlir_fn_res.is_sharded,
        XLA.get_parameter_shardings(exec), # TODO(#2233): use the same workflow as output shardings to parse the tensor sharding attributes directly if possible
        client,
        ndevices,
    )

    concretized_res_names, xla_call_code = codegen_xla_call(
        flatten_arg_names,
        length(linear_results),
        mlir_fn_res.is_sharded,
        ndevices,
        mlir_fn_res.is_pure,
    )

    shard_info_code, optional_shard_info_code, linear_result_shard_info = codegen_shard_info(
        mlir_fn_res.is_sharded,
        length(linear_results),
        linear_results,
        mlir_fn_res.result_shardings,
        exec,
        ndevices,
    )

    unflatten_code, used_shardinfo = codegen_unflatten!(
        linear_args,
        preserved_args,
        concretized_res_names,
        linear_results,
        concrete_result,
        result_stores,
        path_to_shard_info,
        linear_result_shard_info,
        client,
        resharded_inputs,
    )

    for (i, name) in enumerate(linear_result_shard_info)
        if name in used_shardinfo
            push!(shard_info_code, optional_shard_info_code[i])
        end
    end

    sync_call = if compile_options.sync
        calls = []
        for name in concretized_res_names
            push!(calls, :(XLA.synced_buffer($(name))))
        end
        Expr(:block, calls...)
    else
        :()
    end

    donated_buffers_set = if XLA.runtime(client) isa Val{:PJRT}
        :(Base.IdSet{NTuple{<:Any,XLA.PJRT.Buffer}}())
    else
        :(Base.IdSet{XLA.IFRT.Array}())
    end

    body = quote
        global_mesh = $(global_mesh_expr)
        donated_buffers = $(donated_buffers_set)
        donated_args_mask = thunk.donated_args_mask
        $(flatten_code...)
        $(xla_call_code)
        $(sync_call)
        $(shard_info_code...)
        $(unflatten_code...)
        return result
    end

    if DEBUG_PRINT_CODEGEN[] && Reactant.Distributed.local_rank() == 0
        display(body)
        display(mlir_fn_res.donated_args_mask)
    end

    fname = if body in keys(__thunk_rev_body_cache)
        __thunk_rev_body_cache[body]
    else
        fname2 = gensym(Symbol(Symbol(f), :_reactant))
        __thunk_rev_body_cache[body] = fname2
        __thunk_fwd_body_cache[fname2] = body
        fname2
    end

    return register_thunk(
        f,
        Tuple{map(Core.Typeof, args)...},
        body,
        mlir_fn_res.fnwrapped,
        exec,
        mlir_fn_res.is_sharded ? nothing : device,
        str,
        client,
        mlir_fn_res.global_device_ids,
        mlir_fn_res.donated_args_mask,
        compile_options.sync,
    )
end

for cache_type in (:callcache, :sdycache, :sdygroupidcache, :debugcache)
    activate_fn = Symbol(:activate_, cache_type, :!)
    deactivate_fn = Symbol(:deactivate_, cache_type, :!)
    has_fn = Symbol(:_has_, cache_type)

    @eval begin
        function $(activate_fn)(cache)
            stack = get!(task_local_storage(), $(Meta.quot(cache_type))) do
                return []
            end::Vector
            push!(stack, cache)
            return nothing
        end

        function $(deactivate_fn)(cache)
            cache === last(task_local_storage($(Meta.quot(cache_type)))::Vector) ||
                error("Deactivating wrong cache")
            return pop!(task_local_storage($(Meta.quot(cache_type)))::Vector)
        end

        function $(has_fn)()
            return haskey(task_local_storage(), $(Meta.quot(cache_type))) &&
                   !Base.isempty(task_local_storage($(Meta.quot(cache_type)))::Vector)
        end

        function $(cache_type)(; throw_error::Bool=true)
            if !$(has_fn)()
                throw_error && error("No cache is active")
                return nothing
            end
            return last(task_local_storage($(Meta.quot(cache_type)))::Vector)
        end
    end
end

function default_sdycache()
    return Dict{
        Tuple{AbstractVector{Int},NTuple{<:Any,Symbol},Dims{<:Any}},
        @NamedTuple{
            sym_name::MLIR.IR.Attribute,
            mesh_attr::MLIR.IR.Attribute,
            mesh_op::MLIR.IR.Operation,
            mesh::Sharding.Mesh,
        }
    }()
end

mutable struct SdyGroupIDCounter{T}
    @atomic group_id::T
end

function default_sdygroupidcache()
    return SdyGroupIDCounter{Int}(0), Base.IdDict{Union{TracedRArray,TracedRNumber},Int}()
end

function default_callcache()
    return Dict{
        Vector,
        @NamedTuple{
            f_name::String,
            mlir_result_types::Vector{MLIR.IR.Type},
            traced_result::Any,
            mutated_args::Vector{Int},
            linear_results::Vector{Reactant.TracedType},
            fnwrapped::Bool,
            argprefix::Symbol,
            resprefix::Symbol,
            resargprefix::Symbol,
        }
    }()
end

function default_debugcache()
    return Vector{@NamedTuple{f_name::String, file::String, line::Int64}}(undef, 0)
end

# Since we cache these objects we cannot cache data containing MLIR operations (e.g. the entry must be a string
# and not the operation itself).
struct LLVMFunc{F,tt}
    f::Union{F,Nothing}
    entry::String
end

function Base.getproperty(f::LLVMFunc{F,tt}, sym::Symbol) where {F,tt}
    if sym === :fun
        f
    else
        Base.getfield(f, sym)
    end
end

# cache of compilation caches, per module
const _llvm_compiler_caches = Dict{MLIR.IR.Module,Dict{Any,LLVMFunc}}()
function llvm_compiler_cache(_mod::MLIR.IR.Module)
    cache = get(_llvm_compiler_caches, _mod, nothing)
    if cache === nothing
        cache = Dict{Any,LLVMFunc}()
        _llvm_compiler_caches[_mod] = cache
    end
    return cache
end

function clear_llvm_compiler_cache!(_mod::MLIR.IR.Module)
    delete!(_llvm_compiler_caches, _mod)
    return nothing
end

end
