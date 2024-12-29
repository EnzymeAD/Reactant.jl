module ReactantCUDAExt

using CUDA
using Reactant: Reactant, TracedRArray, AnyTracedRArray, MLIR, TracedRNumber
using ReactantCore: @trace
using Libdl

using Adapt

struct CuTracedArray{T,N,A,Size} <: DenseArray{T,N}
    ptr::Core.LLVMPtr{T,A}
end

function Base.show(io::IO, a::AT) where {AT<:CuTracedArray}
    CUDA.Printf.@printf(io, "%s cu traced array at %p", join(size(a), '×'), Int(pointer(a)))
end

## array interface

Base.elsize(::Type{<:CuTracedArray{T}}) where {T} = sizeof(T)
Base.size(g::CuTracedArray{T,N,A,Size}) where {T,N,A,Size} = Size
Base.sizeof(x::CuTracedArray) = Base.elsize(x) * length(x)
function Base.pointer(x::CuTracedArray{T,<:Any,A}) where {T,A}
    return Base.unsafe_convert(Core.LLVMPtr{T,A}, x)
end
@inline function Base.pointer(x::CuTracedArray{T,<:Any,A}, i::Integer) where {T,A}
    return Base.unsafe_convert(Core.LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end

## conversions

function Base.unsafe_convert(
    ::Type{Core.LLVMPtr{T,A}}, x::CuTracedArray{T,<:Any,A}
) where {T,A}
    return x.ptr
end

## indexing intrinsics

CUDA.@device_function @inline function arrayref(
    A::CuTracedArray{T}, index::Integer
) where {T}
    @boundscheck checkbounds(A, index)
    if Base.isbitsunion(T)
        arrayref_union(A, index)
    else
        arrayref_bits(A, index)
    end
end

@inline function arrayref_bits(A::CuTracedArray{T}, index::Integer) where {T}
    return unsafe_load(pointer(A), index)
end

@inline @generated function arrayref_union(
    A::CuTracedArray{T,<:Any,AS}, index::Integer
) where {T,AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel - 1)
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

CUDA.@device_function @inline function arrayset(
    A::CuTracedArray{T}, x::T, index::Integer
) where {T}
    @boundscheck checkbounds(A, index)
    if Base.isbitsunion(T)
        arrayset_union(A, x, index)
    else
        arrayset_bits(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::CuTracedArray{T}, x::T, index::Integer) where {T}
    return unsafe_store!(pointer(A), x, index)
end

@inline @generated function arrayset_union(
    A::CuTracedArray{T,<:Any,AS}, x::T, index::Integer
) where {T,AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel - 1)))

        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(Core.LLVMPtr{$x,AS}, data_ptr), x, 1)
        return nothing
    end
end

CUDA.@device_function @inline function const_arrayref(
    A::CuTracedArray{T}, index::Integer
) where {T}
    @boundscheck checkbounds(A, index)
    return unsafe_cached_load(pointer(A), index)
end

## indexing

Base.IndexStyle(::Type{<:CuTracedArray}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(A::CuTracedArray{T}, i1::Integer) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::CuTracedArray{T}, x, i1::Integer) where {T} =
    arrayset(A, convert(T, x)::T, i1)

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CuTracedArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(
    A::CuTracedArray, I::Union{Integer,CartesianIndex}...
) = A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(
    A::CuTracedArray, x, I::Union{Integer,CartesianIndex}...
) = A[Base._to_linear_index(A, to_indices(A, I)...)] = x

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
        return CuTracedArray{T,N,A}(
            reinterpret(Core.LLVMPtr{T,A}, a.ptr), size(a), a.maxsize
        )
    end

    isize = size(a)
    size1 = div(isize[1] * sizeof(S), sizeof(T))
    osize = tuple(size1, Base.tail(isize)...)
    return CuTracedArray{T,N,A}(reinterpret(Core.LLVMPtr{T,A}, a.ptr), osize, a.maxsize)
end

## reshape

function Base.reshape(a::CuTracedArray{T,M,A}, dims::NTuple{N,Int}) where {T,N,M,A}
    if prod(dims) != length(a)
        throw(
            DimensionMismatch(
                "new dimensions (argument `dims`) must be consistent with array size (`size(a)`)",
            ),
        )
    end
    if N == M && dims == size(a)
        return a
    end
    return _derived_array(a, T, dims)
end

function Adapt.adapt_storage(::CUDA.KernelAdaptor, xs::TracedRArray{T,N}) where {T,N}
    res = CuTracedArray{T,N,CUDA.AS.Global,size(xs)}(
        Base.reinterpret(Core.LLVMPtr{T,CUDA.AS.Global}, Base.pointer_from_objref(xs))
    )
    return res
end

const _kernel_instances = Dict{Any,Any}()

struct LLVMFunc{F,tt}
    f::Union{F,Nothing}
    entry::MLIR.IR.Operation
end

const GPUCompiler = CUDA.GPUCompiler
const LLVM = GPUCompiler.LLVM

function GPULowerCPUFeaturesPass()
    return LLVM.NewPMModulePass("GPULowerCPUFeatures", GPUCompiler.cpu_features!)
end
GPULowerPTLSPass() = LLVM.NewPMModulePass("GPULowerPTLS", GPUCompiler.lower_ptls!)
function GPULowerGCFramePass()
    return LLVM.NewPMFunctionPass("GPULowerGCFrame", GPUCompiler.lower_gc_frame!)
end
function noop_pass(x)
    return false
end
function kern_pass(mod)
    for fname in ("julia.gpu.state_getter",)
        if LLVM.haskey(LLVM.functions(mod), fname)
            fn = LLVM.functions(mod)[fname]
            insts = LLVM.Instruction[]
            for u in LLVM.uses(fn)
                u = LLVM.user(u)
                LLVM.replace_uses!(u, LLVM.UndefValue(LLVM.value_type(u)))
                push!(insts, u)
            end
            for inst in insts
                Reactant.Enzyme.Compiler.eraseInst(LLVM.parent(inst), inst)
            end
            Reactant.Enzyme.Compiler.eraseInst(mod, fn)
        end
    end

    return true
end
AddKernelStatePass() = LLVM.NewPMModulePass("AddKernelStatePass", kern_pass)
LowerKernelStatePass() = LLVM.NewPMFunctionPass("LowerKernelStatePass", noop_pass)
CleanupKernelStatePass() = LLVM.NewPMModulePass("CleanupKernelStatePass", noop_pass)

# compile to executable machine code
function compile(job)
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    entry = GPUCompiler.JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(
            :llvm, job; optimize=false, cleanup=false, validate=false
        )

        entryname = LLVM.name(meta.entry)

        GPUCompiler.optimize_module!(job, mod)
        opt_level = 2
        tm = GPUCompiler.llvm_machine(job.config.target)
        LLVM.@dispose pb = LLVM.NewPMPassBuilder() begin
            LLVM.register!(pb, GPULowerCPUFeaturesPass())
            LLVM.register!(pb, GPULowerPTLSPass())
            LLVM.register!(pb, GPULowerGCFramePass())
            LLVM.register!(pb, AddKernelStatePass())
            LLVM.register!(pb, LowerKernelStatePass())
            LLVM.register!(pb, CleanupKernelStatePass())

            LLVM.add!(pb, LLVM.NewPMModulePassManager()) do mpm
                GPUCompiler.buildNewPMPipeline!(mpm, job, opt_level)
            end
            LLVM.run!(pb, mod, tm)
        end
        GPUCompiler.optimize_module!(job, mod)
        LLVM.run!(CUDA.GPUCompiler.DeadArgumentEliminationPass(), mod, tm)

        for fname in ("gpu_report_exception", "gpu_signal_exception")
            if LLVM.haskey(LLVM.functions(mod), fname)
                fn = LLVM.functions(mod)[fname]
                insts = LLVM.Instruction[]
                for u in LLVM.uses(fn)
                    push!(insts, LLVM.user(u))
                end
                for inst in insts
                    Reactant.Enzyme.Compiler.eraseInst(LLVM.parent(inst), inst)
                end
                Reactant.Enzyme.Compiler.eraseInst(mod, fn)
            end
        end

        LLVM.strip_debuginfo!(mod)
        modstr = string(mod)

        # This is a bit weird since we're taking a module from julia's llvm into reactant's llvm version
        # it is probably safer to reparse a string using the right llvm module api, so we will do that.

        mmod = MLIR.IR.Module(
            @ccall MLIR.API.mlir_c.ConvertLLVMStrToMLIR(
                modstr::Cstring, MLIR.IR.context()::MLIR.API.MlirContext
            )::MLIR.API.MlirModule
        )

        linkRes = @ccall MLIR.API.mlir_c.LinkInModule(
            MLIR.IR.mmodule()::MLIR.API.MlirModule,
            mmod::MLIR.API.MlirModule,
            entryname::Cstring,
        )::MLIR.API.MlirOperation

        entry = MLIR.IR.Operation(linkRes)

        entry
    end
    return LLVMFunc{job.source.specTypes.parameters[1],job.source.specTypes}(nothing, entry)
end

# link into an executable kernel
function link(job, compiled)
    # load as an executable kernel object
    return compiled
end

Reactant.@reactant_overlay @noinline function (func::LLVMFunc{F,tt})(
    args...;
    convert=Val(false),
    blocks::CuDim=1,
    threads::CuDim=1,
    cooperative::Bool=false,
    shmem::Integer=0,
    call_kwargs...,
) where {F,tt}
    blockdim = CUDA.CuDim3(blocks)
    threaddim = CUDA.CuDim3(threads)

    mlir_args = MLIR.IR.Value[]
    restys = MLIR.IR.Type[]
    aliases = MLIR.IR.Attribute[]
    rarrays = TracedRArray[]
    for (i, a) in enumerate(args)
        @assert a isa CuTracedArray
        ta =
            Base.unsafe_pointer_to_objref(Base.reinterpret(Ptr{Cvoid}, a.ptr))::TracedRArray
        push!(rarrays, ta)
        arg = ta.mlir_data
        arg = Reactant.TracedUtils.transpose_val(arg)
        push!(restys, MLIR.IR.type(arg))
        push!(mlir_args, arg)
        push!(
            aliases,
            MLIR.IR.Attribute(
                MLIR.API.stablehloOutputOperandAliasGet(
                    MLIR.IR.context(),
                    length(args) == 1 ? 0 : 1,
                    length(args) == 1 ? C_NULL : Ref{Int64}(i - 1),
                    i - 1,
                    0,
                    C_NULL,
                ),
            ),
        )
    end

    output_operand_aliases = MLIR.IR.Attribute(aliases)

    fname = Reactant.TracedUtils.get_attribute_by_name(func.entry, "sym_name")
    # Force public for now while we don't have real users
    # MLIR.IR.rmattr!(func.entry, "sym_visibility")

    operands = MLIR.IR.Value[]
    for idx in
        (blockdim.x, blockdim.y, blockdim.z, threaddim.x, threaddim.y, threaddim.z, shmem)
        push!(
            operands,
            Reactant.TracedUtils.promote_to(Reactant.TracedRNumber{Int}, idx).mlir_data,
        )
    end
    for arg in mlir_args
        push!(operands, arg)
    end
    owned_regions = MLIR.IR.Region[]
    successors = MLIR.IR.Block[]
    attributes = MLIR.IR.NamedAttribute[
        MLIR.IR.NamedAttribute("fn", MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))),
        MLIR.IR.NamedAttribute(
            "output_operand_aliases", MLIR.IR.Attribute(output_operand_aliases)
        ),
    ]

    location = MLIR.IR.Location()
    call = MLIR.IR.create_operation(
        "enzymexla.kernel_call",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=restys,
        result_inference=false,
    )
    for (i, res) in enumerate(rarrays)
        res.mlir_data = Reactant.TracedUtils.transpose_val(MLIR.IR.result(call, i))
    end
end

# cache of compilation caches, per context
const _compiler_caches = Dict{MLIR.IR.Context,Dict{Any,LLVMFunc}}()
function compiler_cache(ctx::MLIR.IR.Context)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any,LLVMFunc}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

Reactant.@reactant_overlay @noinline function CUDA.cufunction(
    f::F, tt::TT=Tuple{}; kwargs...
) where {F,TT}
    res = Base.@lock CUDA.cufunction_lock begin
        # compile the function
        cache = compiler_cache(MLIR.IR.context())
        source = CUDA.methodinstance(F, tt)
        # cuda = CUDA.active_state()
        device = nothing # cuda.device
        # config = CUDA.compiler_config(device; kwargs...)::CUDA.CUDACompilerConfig
        cuda_cap = v"5.0"
        cuda_ptx = v"6.3"
        llvm_cap = v"5.0"
        llvm_ptx = v"6.3"
        kernel = true
        always_inline = false
        name = nothing
        debuginfo = false
        config = CUDA.CompilerConfig(
            CUDA.PTXCompilerTarget(; cap=llvm_cap, ptx=llvm_ptx, debuginfo),
            CUDA.CUDACompilerParams(; cap=cuda_cap, ptx=cuda_ptx);
            kernel,
            name,
            always_inline,
        )
        CUDA.GPUCompiler.cached_compilation(cache, source, config, compile, link)
    end
    return res
end

function __init__()
    if CUDA.CUDA_Driver_jll.libcuda !== nothing
        handle = Reactant.XLA.Libdl.dlopen(CUDA.CUDA_Driver_jll.libcuda; throw_error=false)
        if handle === nothing
            handle = C_NULL
        end
        ptr1 = Reactant.XLA.Libdl.dlsym(handle, "cuLaunchKernel"; throw_error=false)
        if ptr1 === nothing
            ptr1 = C_NULL
        end
        ptr2 = Reactant.XLA.Libdl.dlsym(handle, "cuModuleLoadData"; throw_error=false)
        if ptr2 === nothing
            ptr2 = C_NULL
        end
        ptr3 = Reactant.XLA.Libdl.dlsym(handle, "cuModuleGetFunction"; throw_error=false)
        if ptr3 === nothing
            ptr3 = C_NULL
        end
        Reactant.Compiler.cuLaunch[] = Base.reinterpret(UInt, ptr1)
        Reactant.Compiler.cuModule[] = Base.reinterpret(UInt, ptr2)
        Reactant.Compiler.cuFunc[] = Base.reinterpret(UInt, ptr3)
    end
    return nothing
end

end # module ReactantCUDAExt
