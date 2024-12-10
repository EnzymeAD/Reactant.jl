module ReactantCUDAExt

using CUDA
using Reactant:
    Reactant, TracedRArray, AnyTracedRArray, materialize_traced_array, MLIR, TracedRNumber
using ReactantCore: @trace

using Adapt

struct CuTracedArray{T,N,A,Size} <: DenseArray{T,N}
    ptr::Core.LLVMPtr{T,A}
end


Base.show(io::IO, a::AT) where AT <: CuTracedArray =
   CUDA.Printf.@printf(io, "%s cu traced array at %p", join(size(a), '×'), Int(pointer(a)))

## array interface

Base.elsize(::Type{<:CuTracedArray{T}}) where {T} = sizeof(T)
Base.size(g::CuTracedArray{T,N,A,Size}) where {T,N,A,Size} = Size
Base.sizeof(x::CuTracedArray) = Base.elsize(x) * length(x)
Base.pointer(x::CuTracedArray{T,<:Any,A}) where {T,A} = Base.unsafe_convert(Core.LLVMPtr{T,A}, x)
@inline function Base.pointer(x::CuTracedArray{T,<:Any,A}, i::Integer) where {T,A}
    Base.unsafe_convert(Core.LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end


## conversions

Base.unsafe_convert(::Type{Core.LLVMPtr{T,A}}, x::CuTracedArray{T,<:Any,A}) where {T,A} =
  x.ptr


## indexing intrinsics

CUDA.@device_function @inline function arrayref(A::CuTracedArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if Base.isbitsunion(T)
        arrayref_union(A, index)
    else
        arrayref_bits(A, index)
    end
end

@inline function arrayref_bits(A::CuTracedArray{T}, index::Integer) where {T}
    unsafe_load(pointer(A), index)
end

@inline @generated function arrayref_union(A::CuTracedArray{T,<:Any,AS}, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel-1)
                ptr = reinterpret(Core.LLVMPtr{$typ,AS}, data_ptr)
		unsafe_load(ptr, 1)
            else
                $ex
            end
        end
    end

    quote
        selector_ptr = typetagdata(A, index)
        selector = unsafe_load(selector_ptr)

        data_ptr = pointer(A, index)

        return $ex
    end
end

CUDA.@device_function @inline function arrayset(A::CuTracedArray{T}, x::T, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if Base.isbitsunion(T)
        arrayset_union(A, x, index)
    else
        arrayset_bits(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::CuTracedArray{T}, x::T, index::Integer) where {T}
    unsafe_store!(pointer(A), x, index)
end

@inline @generated function arrayset_union(A::CuTracedArray{T,<:Any,AS}, x::T, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel-1)))

        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(Core.LLVMPtr{$x,AS}, data_ptr), x, 1)
        return
    end
end

CUDA.@device_function @inline function const_arrayref(A::CuTracedArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    unsafe_cached_load(pointer(A), index)
end


## indexing

Base.IndexStyle(::Type{<:CuTracedArray}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(A::CuTracedArray{T}, i1::Integer) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::CuTracedArray{T}, x, i1::Integer) where {T} =
    arrayset(A, convert(T,x)::T, i1)

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CuTracedArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(A::CuTracedArray,
                                       I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(A::CuTracedArray, x,
                                        I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)] = x


## const indexing

"""
    Const(A::CuTracedArray)

Mark a CuTracedArray as constant/read-only. The invariant guaranteed is that you will not
modify an CuTracedArray for the duration of the current kernel.

This API can only be used on devices with compute capability 3.5 or higher.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct Const{T,N,AS} <: DenseArray{T,N}
    a::CuTracedArray{T,N,AS}
end
Base.Experimental.Const(A::CuTracedArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
Base.axes(C::Const) = axes(C.a)
Base.@propagate_inbounds Base.getindex(A::Const, i1::Integer) = const_arrayref(A.a, i1)

# deprecated
Base.@propagate_inbounds ldg(A::CuTracedArray, i1::Integer) = const_arrayref(A, i1)


## other

@inline function Base.iterate(A::CuTracedArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

function Base.reinterpret(::Type{T}, a::CuTracedArray{S,N,A}) where {T,S,N,A}
  err = GPUArrays._reinterpret_exception(T, a)
  err === nothing || throw(err)

  if sizeof(T) == sizeof(S) # fast case
    return CuTracedArray{T,N,A}(reinterpret(Core.LLVMPtr{T,A}, a.ptr), size(a), a.maxsize)
  end

  isize = size(a)
  size1 = div(isize[1]*sizeof(S), sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)
  return CuTracedArray{T,N,A}(reinterpret(Core.LLVMPtr{T,A}, a.ptr), osize, a.maxsize)
end


## reshape

function Base.reshape(a::CuTracedArray{T,M,A}, dims::NTuple{N,Int}) where {T,N,M,A}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions (argument `dims`) must be consistent with array size (`size(a)`)"))
  end
  if N == M && dims == size(a)
      return a
  end
  _derived_array(a, T, dims)
end



function Adapt.adapt_storage(::CUDA.KernelAdaptor, xs::TracedRArray{T,N}) where {T,N}
  res = CuTracedArray{T,N,CUDA.AS.Global, size(xs)}(Base.reinterpret(Core.LLVMPtr{T,CUDA.AS.Global}, Base.pointer_from_objref(xs)))
  @show res, xs
  return res
end

const _kernel_instances = Dict{Any, Any}()

struct LLVMFunc{F,tt}
   f::Union{F, Nothing}
   mod::String
   image
   entry::String
end


# compile to executable machine code
function compile(job)
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    modstr, image, entry = CUDA.GPUCompiler.JuliaContext() do ctx
            asm, meta = CUDA.GPUCompiler.compile(:asm, job)
	    mod = meta.ir
	    
	    modstr = string(mod)

	    # This is a bit weird since we're taking a module from julia's llvm into reactant's llvm version
	    # it is probably safer to reparse a string using the right llvm module api, so we will do that.

	    mmod = MLIR.IR.Module(@ccall MLIR.API.mlir_c.ConvertLLVMToMLIR(mod::CUDA.LLVM.API.LLVMModuleRef, MLIR.IR.context()::MLIR.API.MlirContext)::MLIR.API.MlirModule)
	    @show mmod

	    # check if we'll need the device runtime
	    undefined_fs = filter(collect(CUDA.LLVM.functions(meta.ir))) do f
		CUDA.LLVM.isdeclaration(f) && !CUDA.LLVM.isintrinsic(f)
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
		!CUDA.isghosttype(dt) && !Core.Compiler.isconstType(dt)
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
			if CUDA.isghosttype(typ) || Core.Compiler.isconstType(typ)
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
	    proc, log = CUDA.run_and_collect(`$(CUDA.ptxas()) $ptxas_opts`)
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
		proc, log = run_and_collect(`$(CUDA.nvlink()) $nvlink_opts`)
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
	    
	    modstr, image, meta.entry
    end
    LLVMFunc{job.source.specTypes.parameters[1],job.source.specTypes}(nothing, modstr, image, CUDA.LLVM.name(entry))
end

# link into an executable kernel
function link(job, compiled)
    # load as an executable kernel object
    return compiled
end

function transpose_val(val)
    attr = MLIR.IR.DenseArrayAttribute(
        Int64[reverse(0:(length(size(MLIR.IR.type(val))) - 1))...]
    )
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val; permutation=attr), 1)
end

function (func::LLVMFunc{F,tt})(args...; convert=Val(false), blocks::CuDim=1, threads::CuDim=1,
                cooperative::Bool=false, shmem::Integer=0, call_kwargs...) where{F, tt}
    @show args
    @show call_kwargs

    blockdim = CUDA.CuDim3(blocks)
    threaddim = CUDA.CuDim3(threads)

    mlir_args = MLIR.IR.Value[]
    restys = MLIR.IR.Type[]
    aliases = MLIR.IR.Attribute[]
    rarrays = TracedRArray[]
    for (i, a) in enumerate(args)
	@show a
	@assert a isa CuTracedArray
	ta = Base.unsafe_pointer_to_objref(Base.reinterpret(Ptr{Cvoid}, a.ptr))::TracedRArray
	push!(rarrays, ta)
	arg = ta.mlir_data
	arg = transpose_val(arg)
	@show arg
	push!(restys, MLIR.IR.type(arg))
	push!(mlir_args, arg)
	push!(aliases,
	      MLIR.IR.Attribute(MLIR.API.stablehloOutputOperandAliasGet(
		MLIR.IR.context(),
	        length(args) == 1 ? 0 : 1,
		length(args) == 1 ? C_NULL : Ref{Int64}(i-1),
		i-1,
		0,
		C_NULL
		))
	      )
    end

    output_operand_aliases=MLIR.IR.Attribute(aliases)
    call = MLIR.Dialects.stablehlo.custom_call(mlir_args; result_0=restys, call_target_name="reactant_gpu_call", output_operand_aliases, backend_config=MLIR.IR.Attribute("configstr"))
    # call = MLIR.Dialects.stablehlo.custom_call(mlir_args; result_0=restys, call_target_name="reactant_gpu_call", output_operand_aliases, backend_config=MLIR.IR.Attribute(func.mod))
    for (i, res) in enumerate(rarrays)
       res.mlir_data = transpose_val(MLIR.IR.result(call, i))
    end
    #CUDA.cuLaunchKernel(f,
    #	       blockdim.x, blockdim.y, blockdim.z,
    #	       threaddim.x, threaddim.y, threaddim.z,
    #	       shmem, stream, kernelParams, C_NULL)  
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

Reactant.@reactant_override function CUDA.cufunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    @show "recufunction", f, tt
    res = Base.@lock CUDA.cufunction_lock begin
        # compile the function
	cache = compiler_cache(MLIR.IR.context())
        source = CUDA.methodinstance(F, tt)

    	cuda = CUDA.active_state()
        config = CUDA.compiler_config(cuda.device; kwargs...)::CUDA.CUDACompilerConfig
        CUDA.GPUCompiler.cached_compilation(cache, source, config, compile, link)
    end
    res
end

function __init__()
   
end

end # module ReactantCUDAExt
