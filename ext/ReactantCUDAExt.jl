module ReactantCUDAExt

using CUDA
using Reactant:
    Reactant, TracedRArray, AnyTracedRArray, materialize_traced_array, MLIR, TracedRNumber
using ReactantCore: @trace

using Adapt

function Adapt.adapt_storage(::CUDA.KernelAdaptor, xs::TracedRArray{T,N}) where {T,N}
  res = CuDeviceArray{T,N,CUDA.AS.Global}(Base.reinterpret(Core.LLVMPtr{T,CUDA.AS.Global}, xs.mlir_data.value.ptr), size(xs))
  @show res, xs
  return res
end

const _kernel_instances = Dict{Any, Any}()


# compile to executable machine code
function compile(job)
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    modstr = CUDA.GPUCompiler.JuliaContext() do ctx
        mod, meta = CUDA.GPUCompiler.compile(:llvm, job)
	string(mod)
    end
    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(meta.ir))) do f
        isdeclaration(f) && !CUDA.LLVM.isintrinsic(f)
    end
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                     "__nvvm_reflect" #= TODO: should have been optimized away =#]
    needs_cudadevrt = !isempty(setdiff(CUDA.LLVM.name.(undefined_fs), intrinsic_fns))

    # prepare invocations of CUDA compiler tools
    ptxas_opts = String[]
    nvlink_opts = String[]
    ## debug flags
    if Base.JLOptions().debug_level == 1
        push!(ptxas_opts, "--generate-line-info")
    elseif Base.JLOptions().debug_level >= 2
        push!(ptxas_opts, "--device-debug")
        push!(nvlink_opts, "--debug")
    end
    ## relocatable device code
    if needs_cudadevrt
        push!(ptxas_opts, "--compile-only")
    end

    ptx = job.config.params.ptx
    cap = job.config.params.cap
    arch = "sm_$(cap.major)$(cap.minor)"

    # validate use of parameter memory
    argtypes = filter([CUDA.KernelState, job.source.specTypes.parameters...]) do dt
        !isghosttype(dt) && !Core.Compiler.isconstType(dt)
    end
    param_usage = sum(sizeof, argtypes)
    param_limit = 4096
    if cap >= v"7.0" && ptx >= v"8.1"
        param_limit = 32764
    end
    if param_usage > param_limit
        msg = """Kernel invocation uses too much parameter memory.
                 $(Base.format_bytes(param_usage)) exceeds the $(Base.format_bytes(param_limit)) limit imposed by sm_$(cap.major)$(cap.minor) / PTX v$(ptx.major).$(ptx.minor)."""

        try
            details = "\n\nRelevant parameters:"

            source_types = job.source.specTypes.parameters
            source_argnames = Base.method_argnames(job.source.def)
            while length(source_argnames) < length(source_types)
                # this is probably due to a trailing vararg; repeat its name
                push!(source_argnames, source_argnames[end])
            end

            for (i, typ) in enumerate(source_types)
                if isghosttype(typ) || Core.Compiler.isconstType(typ)
                    continue
                end
                name = source_argnames[i]
                details *= "\n  [$(i-1)] $name::$typ uses $(Base.format_bytes(sizeof(typ)))"
            end
            details *= "\n"

            if cap >= v"7.0" && ptx < v"8.1" && param_usage < 32764
                details *= "\nNote: use a newer CUDA to support more parameters on your device.\n"
            end

            msg *= details
        catch err
            @error "Failed to analyze kernel parameter usage; please file an issue with a reproducer."
        end
        error(msg)
    end

    # compile to machine code
    # NOTE: we use tempname since mktemp doesn't support suffixes, and mktempdir is slow
    ptx_input = tempname(cleanup=false) * ".ptx"
    ptxas_output = tempname(cleanup=false) * ".cubin"
    write(ptx_input, asm)

    # we could use the driver's embedded JIT compiler, but that has several disadvantages:
    # 1. fixes and improvements are slower to arrive, by using `ptxas` we only need to
    #    upgrade the toolkit to get a newer compiler;
    # 2. version checking is simpler, we otherwise need to use NVML to query the driver
    #    version, which is hard to correlate to PTX JIT improvements;
    # 3. if we want to be able to use newer (minor upgrades) of the CUDA toolkit on an
    #    older driver, we should use the newer compiler to ensure compatibility.
    append!(ptxas_opts, [
        "--verbose",
        "--gpu-name", arch,
        "--output-file", ptxas_output,
        ptx_input
    ])
    proc, log = CUDA.run_and_collect(`$(ptxas()) $ptxas_opts`)
    log = strip(log)
    if !success(proc)
        reason = proc.termsignal > 0 ? "ptxas received signal $(proc.termsignal)" :
                                       "ptxas exited with code $(proc.exitcode)"
        msg = "Failed to compile PTX code ($reason)"
        msg *= "\nInvocation arguments: $(join(ptxas_opts, ' '))"
        if !isempty(log)
            msg *= "\n" * log
        end
        msg *= "\nIf you think this is a bug, please file an issue and attach $(ptx_input)"
        if parse(Bool, get(ENV, "BUILDKITE", "false"))
            run(`buildkite-agent artifact upload $(ptx_input)`)
        end
        error(msg)
    elseif !isempty(log)
        @debug "PTX compiler log:\n" * log
    end
    rm(ptx_input)
    
    # link device libraries, if necessary
    #
    # this requires relocatable device code, which prevents certain optimizations and
    # hurts performance. as such, we only do so when absolutely necessary.
    # TODO: try LTO, `--link-time-opt --nvvmpath /opt/cuda/nvvm`.
    #       fails with `Ignoring -lto option because no LTO objects found`
    if needs_cudadevrt
        nvlink_output = tempname(cleanup=false) * ".cubin"
        append!(nvlink_opts, [
            "--verbose", "--extra-warnings",
            "--arch", arch,
            "--library-path", dirname(libcudadevrt),
            "--library", "cudadevrt",
            "--output-file", nvlink_output,
            ptxas_output
        ])
        proc, log = run_and_collect(`$(nvlink()) $nvlink_opts`)
        log = strip(log)
        if !success(proc)
            reason = proc.termsignal > 0 ? "nvlink received signal $(proc.termsignal)" :
                                           "nvlink exited with code $(proc.exitcode)"
            msg = "Failed to link PTX code ($reason)"
            msg *= "\nInvocation arguments: $(join(nvlink_opts, ' '))"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(ptxas_output)"
            error(msg)
        elseif !isempty(log)
            @debug "PTX linker info log:\n" * log
        end
        rm(ptxas_output)

        image = read(nvlink_output)
        rm(nvlink_output)
    else
        image = read(ptxas_output)
        rm(ptxas_output)
    end
    
    println(string(modstr))
    @show job
    @show job.source
    @show job.config
    LLVMFunc{F,job.source.specTypes}(f, modstr, image, LLVM.name(meta.entry))
end

# link into an executable kernel
function link(job, compiled)
    # load as an executable kernel object
    return compiled
end

struct LLVMFunc{F,tt}
   f::F
   mod::String
   image
   entry::String
end

function (func::LLVMFunc{F,tt})(args...; blocks::CUDA.CuDim=1, threads::CUDA.CuDim=1,
					 shmem::Integer=0) where{F, tt}
    blockdim = CUDA.CuDim3(blocks)
    threaddim = CUDA.CuDim3(threads)

    @show args

# void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
#                         size_t opaque_len, XlaCustomCallStatus* status) {

    CUDA.cuLaunchKernel(f,
	       blockdim.x, blockdim.y, blockdim.z,
	       threaddim.x, threaddim.y, threaddim.z,
	       shmem, stream, kernelParams, C_NULL)  
end

# cache of compilation caches, per context
const _compiler_caches = Dict{MLIR.IR.Context, Dict{Any, LLVMFunc}}();
function compiler_cache(ctx::MLIR.IR.Context)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, LLVMFunc}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

function recufunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    cuda = CUDA.active_state()
    @show f, tt
    flush(stdout)

    Base.@lock CUDA.cufunction_lock begin
        # compile the function
	cache = compiler_cache(MLIR.IR.context())
        source = CUDA.methodinstance(F, tt)
        config = CUDA.compiler_config(cuda.device; kwargs...)::CUDA.CUDACompilerConfig
        fun = CUDA.GPUCompiler.cached_compilation(cache, source, config, compile, link)

	#@show fun.mod
	# create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        key = (objectid(source))
        kernel = get(_kernel_instances, key, nothing)
        if kernel === nothing
            _kernel_instances[key] = kernel
        end
        return kernel::LLVMFunc{F,tt}
    end
end

const CC = Core.Compiler

import Core.Compiler:
    AbstractInterpreter,
    abstract_call,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
    get_max_methods,
    CallMeta,
    Effects,
    NoCallInfo,
    widenconst,
    mapany,
    MethodResultPure


function Reactant.set_reactant_abi(
    interp,
    f::typeof(CUDA.cufunction),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int=get_max_methods(interp, f, sv),
)
    (; fargs, argtypes) = arginfo
	
	arginfo2 = ArgInfo(
	    if fargs isa Nothing
		nothing
	    else
		[:($(recufunction)), fargs[2:end]...]
	    end,
	    [Core.Const(recufunction), argtypes[2:end]...],
	)
	return abstract_call_known(interp, recufunction, arginfo2, si, sv, max_methods)
end

end # module ReactantCUDAExt
