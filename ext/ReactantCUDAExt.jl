module ReactantCUDAExt

using Reactant: Reactant, TracedRArray, AnyConcretePJRTArray, MLIR, TracedRNumber
using Reactant.Compiler: raising
using Reactant.Ops: @opcall

using Adapt: Adapt, adapt
using CUDA: CUDA, CuDim, DenseCuArray, unsafe_cached_load

using GPUCompiler: GPUCompiler
using KernelAbstractions: KernelAbstractions
using LLVM: LLVM

using PrecompileTools: @setup_workload, @compile_workload

const KA = KernelAbstractions

Reactant.is_extension_loaded(::Val{:CUDA}) = true

struct CuTracedArray{T,N,A,Size} <: DenseArray{T,N}
    ptr::Core.LLVMPtr{T,A}

    function CuTracedArray{T,N,A,Size}(xs::TracedRArray) where {T,N,A,Size}
        gc_vec = Reactant.Compiler.context_gc_vector[MLIR.IR.context()]
        push!(gc_vec, xs)
        @assert gc_vec[end] === xs
        ptr = Base.reinterpret(Core.LLVMPtr{T,CUDA.AS.Global}, Base.pointer_from_objref(xs))
        return new(ptr)
    end
end

Reactant.use_overlayed_version(::CuTracedArray) = true

struct CuTracedRNumber{T,A} <: Number
    ptr::Core.LLVMPtr{T,A}

    function CuTracedRNumber{T,A}(xs::TracedRNumber) where {T,A}
        gc_vec = Reactant.Compiler.context_gc_vector[MLIR.IR.context()]
        push!(gc_vec, xs)
        @assert gc_vec[end] === xs
        ptr = Base.reinterpret(Core.LLVMPtr{T,CUDA.AS.Global}, Base.pointer_from_objref(xs))
        return new(ptr)
    end
    function CuTracedRNumber{T,A}(ptr::Core.LLVMPtr{T,A}) where {T,A}
        return new(ptr)
    end
end

Reactant.use_overlayed_version(::CuTracedRNumber) = true

Base.@nospecializeinfer Reactant.is_traced_number(
    @nospecialize(T::Type{<:CuTracedRNumber})
) = true
Reactant.unwrapped_eltype(::Type{<:CuTracedRNumber{T}}) where {T} = T

@inline CuTracedRNumber{T,A}(val::Number) where {T,A} = convert(CuTracedRNumber{T,A}, val)

function Base.getindex(RN::CuTracedRNumber{T,A}) where {T,A}
    align = alignment(RN)
    return @inbounds unsafe_load(RN.ptr, 1, Val(align))
end

Base.convert(::Type{T}, RN::CuTracedRNumber) where {T<:Number} = convert(T, getindex(RN))

for jlop in (
    :(Base.min),
    :(Base.mod),
    :(Base.max),
    :(Base.:+),
    :(Base.:-),
    :(Base.:*),
    :(Base.:/),
    :(Base.:^),
    :(Base.rem),
    :(Base.isless),
    :(Base.:(==)),
    :(Base.:(!=)),
)
    @eval begin
        @inline $jlop(a::CuTracedRNumber, b::CuTracedRNumber) = $jlop(a[], b[])
        @inline $jlop(a::CuTracedRNumber{T,A}, b::Number) where {T,A} = $jlop(a[], b)
        @inline $jlop(a::Number, b::CuTracedRNumber{T,A}) where {T,A} = $jlop(a, b[])
    end
end

@inline Base.ifelse(cond::Bool, a, b::CuTracedRNumber) = ifelse(cond, a, b[])
@inline Base.ifelse(cond::Bool, a::CuTracedRNumber, b) = ifelse(cond, a[], b)
@inline Base.ifelse(cond::Bool, a::CuTracedRNumber, b::CuTracedRNumber) =
    ifelse(cond, a[], b[])
@inline Base.ifelse(cond::CuTracedRNumber, a, b) = ifelse(cond[], a, b)
@inline Base.ifelse(cond::CuTracedRNumber, a::CuTracedRNumber, b) = ifelse(cond[], a[], b)
@inline Base.ifelse(cond::CuTracedRNumber, a, b::CuTracedRNumber) = ifelse(cond[], a, b[])
@inline Base.ifelse(cond::CuTracedRNumber, a::CuTracedRNumber, b::CuTracedRNumber) =
    ifelse(cond[], a[], b[])

Base.@constprop :aggressive @inline Base.:^(
    a::CuTracedRNumber{T,A}, b::Integer
) where {T,A} = ^(a[], b)

@inline Base.unsafe_trunc(::Type{T}, a::CuTracedRNumber) where {T} =
    Base.unsafe_trunc(T, a[])

for jlop in (:(Base.:+), :(Base.:-), :(Base.isnan), :(Base.isfinite), :(Base.isinf))
    @eval begin
        @inline $jlop(a::CuTracedRNumber) = $jlop(a[])
    end
end

Base.OneTo(x::CuTracedRNumber{<:Integer}) = Base.OneTo(x[])

@static if isdefined(Base, :unchecked_oneto)
    function Base.unchecked_oneto(x::CuTracedRNumber{<:Integer})
        return Base.unchecked_oneto(x[])
    end
end

@inline function Base.convert(CT::Type{CuTracedRNumber{Float64,1}}, x::Number)
    return CT(
        Base.reinterpret(
            Core.LLVMPtr{Float64,1},
            Base.llvmcall(
                (
                    """define i8 addrspace(1)* @entry(double %d) alwaysinline {
              %a = alloca double
              store atomic double %d, double* %a release, align 8
       %bc = bitcast double* %a to i8*
              %ac = addrspacecast i8* %bc to i8 addrspace(1)*
              ret i8 addrspace(1)* %ac
                        }
          """,
                    "entry",
                ),
                Core.LLVMPtr{UInt8,1},
                Tuple{Float64},
                convert(Float64, x),
            ),
        ),
    )
end

@inline function Base.convert(CT::Type{CuTracedRNumber{Float32,1}}, x::Number)
    return CT(
        Base.reinterpret(
            Core.LLVMPtr{Float32,1},
            Base.llvmcall(
                (
                    """define i8 addrspace(1)* @entry(float %d) alwaysinline {
              %a = alloca float
              store atomic float %d, float* %a release, align 4
       %bc = bitcast float* %a to i8*
              %ac = addrspacecast i8* %bc to i8 addrspace(1)*
              ret i8 addrspace(1)* %ac
                        }
          """,
                    "entry",
                ),
                Core.LLVMPtr{UInt8,1},
                Tuple{Float32},
                convert(Float32, x),
            ),
        ),
    )
end

Base.convert(::Type{<:CuTracedRNumber{T}}, x::CuTracedRNumber{T}) where {T} = x

Base.one(a::CuTracedRNumber) = one(a[])
Base.one(::Type{<:CuTracedRNumber{T,A}}) where {T,A} = one(T)
Base.zero(a::CuTracedRNumber) = zero(a[])
Base.zero(::Type{<:CuTracedRNumber{T,A}}) where {T,A} = zero(T)

Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber{T}}),
    @nospecialize(b::Type{<:CuTracedRNumber{T2}})
) where {T,T2}
    return promote_rule(T, T2)
end
Base.@nospecializeinfer function Base.promote_rule(
    ::Type{Any}, @nospecialize(b::Type{<:CuTracedRNumber})
)
    return Any
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber}), ::Type{Any}
)
    return Any
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(T2::Type), @nospecialize(b::Type{<:CuTracedRNumber{T}})
) where {T}
    if T == T2
        return T
    else
        return promote_rule(T, T2)
    end
end
Base.@nospecializeinfer function Base.promote_rule(
    @nospecialize(a::Type{<:CuTracedRNumber{T}}), @nospecialize(T2::Type)
) where {T}
    if T == T2
        return T
    else
        return promote_rule(T, T2)
    end
end

Base.@nospecializeinfer function Reactant.promote_traced_type(
    @nospecialize(a::Type{<:CuTracedRNumber{T,A}}),
    @nospecialize(b::Type{<:CuTracedRNumber{T2,A}})
) where {T,T2,A}
    return CuTracedRNumber{Reactant.promote_traced_type(T, T2),A}
end
Base.@nospecializeinfer function Reactant.promote_traced_type(
    ::Type{Any}, @nospecialize(b::Type{<:CuTracedRNumber})
)
    return Any
end
Base.@nospecializeinfer function Reactant.promote_traced_type(
    @nospecialize(a::Type{<:CuTracedRNumber}), ::Type{Any}
)
    return Any
end
Base.@nospecializeinfer function Reactant.promote_traced_type(
    @nospecialize(T2::Type), ::Type{<:CuTracedRNumber{T,A}}
) where {T,A}
    if T == T2
        return CuTracedRNumber{T,A}
    else
        return CuTracedRNumber{Reactant.promote_trace_type(T, T2),A}
    end
end
Base.@nospecializeinfer function Reactant.promote_traced_type(
    ::Type{<:CuTracedRNumber{T,A}}, @nospecialize(T2::Type)
) where {T,A}
    if T == T2
        return CuTracedRNumber{T,A}
    else
        return CuTracedRNumber{Reactant.promote_trace_type(T, T2),A}
    end
end

function Base.show(io::IO, a::AT) where {AT<:CuTracedArray}
    CUDA.Printf.@printf(io, "%s cu traced array at %p", join(size(a), '×'), Int(pointer(a)))
end

function Base.show(io::IO, a::AT) where {AT<:CuTracedRNumber}
    CUDA.Printf.@printf(
        io, "%s cu traced rnumber at %p", join(size(a), '×'), Int(pointer(a))
    )
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

# TODO: arrays as allocated by the CUDA APIs are 256-byte aligned. we should keep track of
#       this information, because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@generated function alignment(::CuTracedArray{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end
@generated function alignment(::CuTracedRNumber{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
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
    align = alignment(A)
    return unsafe_load(pointer(A), index, Val(align))
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
    align = alignment(A)
    return unsafe_store!(pointer(A), x, index, Val(align))
end

@inline @generated function arrayset_union(
    A::CuTracedArray{T,<:Any,AS}, x::T, index::Integer
) where {T,AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel - 1)))

        align = alignment(A)
        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(Core.LLVMPtr{$x,AS}, data_ptr), x, 1, Val(align))
        return nothing
    end
end

CUDA.@device_function @inline function const_arrayref(
    A::CuTracedArray{T}, index::Integer
) where {T}
    @boundscheck checkbounds(A, index)
    align = alignment(A)
    return unsafe_cached_load(pointer(A), index, Val(align))
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

struct ReactantKernelAdaptor end

function Adapt.adapt_storage(to::ReactantKernelAdaptor, p::CUDA.CuPtr)
    return error("Cannot convert CuPtr argument of Reactant Kernel")
end
function Adapt.adapt_storage(ka::ReactantKernelAdaptor, xs::DenseCuArray)
    return Adapt.adapt_storage(ka, Array(xs))
end
function Adapt.adapt_storage(ka::ReactantKernelAdaptor, xs::Array)
    return Adapt.adapt_storage(ka, @opcall(constant(xs)))
end
function Adapt.adapt_structure(
    to::ReactantKernelAdaptor, bc::Broadcast.Broadcasted{Style,<:Any,Type{T}}
) where {Style,T}
    return Broadcast.Broadcasted{Style}(
        (x...) -> T(x...), Adapt.adapt(to, bc.args), bc.axes
    )
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function Reactant.ka_with_reactant(ndrange, workgroupsize, obj, args...)
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(
        obj, ndrange, workgroupsize
    )
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = CUDA.@cuda launch = false always_inline = backend.always_inline maxthreads =
        maxthreads obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        if !Reactant.Compiler.PartitionKA[] || raising() || !CUDA.functional()
            threads = prod(ndrange)
        else
            config = CUDA.launch_configuration(kernel.fun; max_threads=prod(ndrange))
            if backend.prefer_blocks
                # Prefer blocks over threads
                threads = min(prod(ndrange), config.threads)
                # XXX: Some kernels performs much better with all blocks active
                cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
                threads = cld(prod(ndrange), cu_blocks)
            else
                threads = config.threads
            end
            workgroupsize = threads_to_workgroupsize(threads, ndrange)
            iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        end
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if blocks == 0
        return nothing
    end

    # Launch kernel
    kernel(ctx, args...; threads, blocks)

    return nothing
end

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuTracedArray) = Base.Experimental.Const(a)

struct ReactantRefValue{T} <: Ref{T}
    val::T
end
Base.getindex(r::ReactantRefValue{T}) where {T} = r.val
function Adapt.adapt_structure(to::ReactantKernelAdaptor, ref::Base.RefValue)
    return ReactantRefValue(adapt(to, ref[]))
end

function recudaconvert(arg)
    return adapt(ReactantKernelAdaptor(), arg)
end
Reactant.@reactant_overlay @noinline function CUDA.cudaconvert(arg)
    return recudaconvert(arg)
end

function Adapt.adapt_storage(::ReactantKernelAdaptor, xs::TracedRArray{T,N}) where {T,N}
    res = CuTracedArray{T,N,CUDA.AS.Global,size(xs)}(xs)
    return res
end

function Adapt.adapt_storage(::ReactantKernelAdaptor, xs::TracedRNumber{T}) where {T}
    res = CuTracedRNumber{T,CUDA.AS.Global}(xs)
    return res
end

import Reactant.TracedStepRangeLen

function Adapt.adapt_storage(::ReactantKernelAdaptor, r::TracedStepRangeLen)
    return TracedStepRangeLen(
        Adapt.adapt(ReactantKernelAdaptor(), r.ref),
        Adapt.adapt(ReactantKernelAdaptor(), r.step),
        Adapt.adapt(ReactantKernelAdaptor(), r.len),
        Adapt.adapt(ReactantKernelAdaptor(), r.offset),
    )
end

function Adapt.adapt_storage(::ReactantKernelAdaptor, r::Base.TwicePrecision)
    return Base.TwicePrecision(
        Adapt.adapt(ReactantKernelAdaptor(), r.hi),
        Adapt.adapt(ReactantKernelAdaptor(), r.lo),
    )
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

# TODO in the future we may want to avoid doing a second cufunction compilation
# for computing the thread/block count (or potentially do it ourselves).
@noinline function CUDA.launch_configuration(
    f::LLVMFunc{F,tt}; shmem::Union{Integer,Base.Callable}=0, max_threads::Integer=0
) where {F,tt}
    return CUDA.launch_configuration(
        Base.inferencebarrier(CUDA.cufunction)(f.f, Tuple{tt.parameters[2:end]...}).fun;
        shmem,
        max_threads,
    )
end

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

# From https://github.com/JuliaGPU/GPUCompiler.jl/blob/7b9322faa34685026c4601a5084eecf5a5d7f3fe/src/ptx.jl#L149
function vendored_optimize_module!(
    @nospecialize(job), mod::LLVM.Module, instcombine::Bool=false
)
    tm = GPUCompiler.llvm_machine(job.config.target)
    # TODO: Use the registered target passes (JuliaGPU/GPUCompiler.jl#450)
    LLVM.@dispose pb = LLVM.NewPMPassBuilder() begin
        LLVM.register!(pb, GPUCompiler.NVVMReflectPass())

        LLVM.add!(pb, LLVM.NewPMFunctionPassManager()) do fpm
            # TODO: need to run this earlier; optimize_module! is called after addOptimizationPasses!
            LLVM.add!(fpm, GPUCompiler.NVVMReflectPass())

            # needed by GemmKernels.jl-like code
            LLVM.add!(fpm, LLVM.SpeculativeExecutionPass())

            # NVPTX's target machine info enables runtime unrolling,
            # but Julia's pass sequence only invokes the simple unroller.
            LLVM.add!(fpm, LLVM.LoopUnrollPass(; job.config.opt_level))
            if instcombine
                LLVM.add!(fpm, LLVM.InstCombinePass())        # clean-up redundancy
            else
                LLVM.add!(fpm, LLVM.InstSimplifyPass())        # clean-up redundancy
            end
            LLVM.add!(fpm, LLVM.NewPMLoopPassManager(; use_memory_ssa=true)) do lpm
                LLVM.add!(lpm, LLVM.LICMPass())           # the inner runtime check might be outer loop invariant
            end

            # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
            # that code still needs to be optimized (arguably, multiple unroll passes should be
            # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
            if job.config.opt_level == 2
                LLVM.add!(fpm, LLVM.GVNPass())
            elseif job.config.opt_level == 1
                LLVM.add!(fpm, LLVM.EarlyCSEPass())
            end
            LLVM.add!(fpm, LLVM.DSEPass())

            LLVM.add!(fpm, LLVM.SimplifyCFGPass())
        end

        # get rid of the internalized functions; now possible unused
        LLVM.add!(pb, LLVM.GlobalDCEPass())

        LLVM.run!(pb, mod, tm)
    end
end

function vendored_buildEarlyOptimizerPipeline(
    mpm, @nospecialize(job), opt_level; instcombine=false
)
    LLVM.add!(mpm, LLVM.NewPMCGSCCPassManager()) do cgpm
        # TODO invokeCGSCCCallbacks
        LLVM.add!(cgpm, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.Interop.AllocOptPass())
            LLVM.add!(fpm, LLVM.Float2IntPass())
            LLVM.add!(fpm, LLVM.LowerConstantIntrinsicsPass())
        end
    end
    LLVM.add!(mpm, GPULowerCPUFeaturesPass())
    if opt_level >= 1
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            if opt_level >= 2
                LLVM.add!(fpm, LLVM.SROAPass())
                if instcombine
                    LLVM.add!(fpm, LLVM.InstCombinePass())
                else
                    LLVM.add!(fpm, LLVM.InstSimplifyPass())
                end
                LLVM.add!(fpm, LLVM.JumpThreadingPass())
                LLVM.add!(fpm, LLVM.CorrelatedValuePropagationPass())
                LLVM.add!(fpm, LLVM.ReassociatePass())
                LLVM.add!(fpm, LLVM.EarlyCSEPass())
                LLVM.add!(fpm, LLVM.Interop.AllocOptPass())
            else
                if instcombine
                    LLVM.add!(fpm, LLVM.InstCombinePass())
                else
                    LLVM.add!(fpm, LLVM.InstSimplifyPass())
                end
                LLVM.add!(fpm, LLVM.EarlyCSEPass())
            end
        end
        # TODO invokePeepholeCallbacks
    end
end

function vendored_buildIntrinsicLoweringPipeline(
    mpm, @nospecialize(job), opt_level; instcombine::Bool=false
)
    GPUCompiler.add!(mpm, LLVM.Interop.RemoveNIPass())

    # lower GC intrinsics
    if !GPUCompiler.uses_julia_runtime(job)
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, GPULowerGCFramePass())
        end
    end

    # lower kernel state intrinsics
    # NOTE: we can only do so here, as GC lowering can introduce calls to the runtime,
    #       and thus additional uses of the kernel state intrinsics.
    if job.config.kernel
        # TODO: now that all kernel state-related passes are being run here, merge some?
        LLVM.add!(mpm, AddKernelStatePass())
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LowerKernelStatePass())
        end
        LLVM.add!(mpm, CleanupKernelStatePass())
    end

    if !GPUCompiler.uses_julia_runtime(job)
        # remove dead uses of ptls
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.ADCEPass())
        end
        LLVM.add!(mpm, GPULowerPTLSPass())
    end

    LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
        # lower exception handling
        if GPUCompiler.uses_julia_runtime(job)
            LLVM.add!(fpm, LLVM.Interop.LowerExcHandlersPass())
        end
        LLVM.add!(fpm, GPUCompiler.GCInvariantVerifierPass())
        LLVM.add!(fpm, LLVM.Interop.LateLowerGCPass())
        if GPUCompiler.uses_julia_runtime(job) && VERSION >= v"1.11.0-DEV.208"
            LLVM.add!(fpm, LLVM.Interop.FinalLowerGCPass())
        end
    end
    if GPUCompiler.uses_julia_runtime(job) && VERSION < v"1.11.0-DEV.208"
        LLVM.add!(mpm, LLVM.Interop.FinalLowerGCPass())
    end

    if opt_level >= 2
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.GVNPass())
            LLVM.add!(fpm, LLVM.SCCPPass())
            LLVM.add!(fpm, LLVM.DCEPass())
        end
    end

    # lower PTLS intrinsics
    if GPUCompiler.uses_julia_runtime(job)
        LLVM.add!(mpm, LLVM.Interop.LowerPTLSPass())
    end

    if opt_level >= 1
        LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
            if instcombine
                LLVM.add!(fpm, LLVM.InstCombinePass())
            else
                LLVM.add!(fpm, LLVM.InstSimplifyPass())
            end
            LLVM.add!(
                fpm, LLVM.SimplifyCFGPass(; GPUCompiler.AggressiveSimplifyCFGOptions...)
            )
        end
    end

    # remove Julia address spaces
    LLVM.add!(mpm, LLVM.Interop.RemoveJuliaAddrspacesPass())

    # Julia's operand bundles confuse the inliner, so repeat here now they are gone.
    # FIXME: we should fix the inliner so that inlined code gets optimized early-on
    return LLVM.add!(mpm, LLVM.AlwaysInlinerPass())
end

function vendored_buildScalarOptimizerPipeline(
    fpm, @nospecialize(job), opt_level; instcombine::Bool=false
)
    if opt_level >= 2
        LLVM.add!(fpm, LLVM.Interop.AllocOptPass())
        LLVM.add!(fpm, LLVM.SROAPass())
        LLVM.add!(fpm, LLVM.InstSimplifyPass())
        LLVM.add!(fpm, LLVM.GVNPass())
        LLVM.add!(fpm, LLVM.MemCpyOptPass())
        LLVM.add!(fpm, LLVM.SCCPPass())
        LLVM.add!(fpm, LLVM.CorrelatedValuePropagationPass())
        LLVM.add!(fpm, LLVM.DCEPass())
        LLVM.add!(fpm, LLVM.IRCEPass())
        if instcombine
            LLVM.add!(fpm, LLVM.InstCombinePass())
        else
            LLVM.add!(fpm, LLVM.InstSimplifyPass())
        end
        LLVM.add!(fpm, LLVM.JumpThreadingPass())
    end
    if opt_level >= 3
        LLVM.add!(fpm, LLVM.GVNPass())
    end
    if opt_level >= 2
        LLVM.add!(fpm, LLVM.DSEPass())
        # TODO invokePeepholeCallbacks
        LLVM.add!(fpm, LLVM.SimplifyCFGPass(; GPUCompiler.AggressiveSimplifyCFGOptions...))
        LLVM.add!(fpm, LLVM.Interop.AllocOptPass())
        LLVM.add!(fpm, LLVM.NewPMLoopPassManager()) do lpm
            LLVM.add!(lpm, LLVM.LoopDeletionPass())
            LLVM.add!(lpm, LLVM.LoopInstSimplifyPass())
        end
        LLVM.add!(fpm, LLVM.LoopDistributePass())
    end
    # TODO invokeScalarOptimizerCallbacks
end

function vendored_buildNewPMPipeline!(mpm, @nospecialize(job), opt_level)
    # Doesn't call instcombine
    GPUCompiler.buildEarlySimplificationPipeline(mpm, job, opt_level)
    LLVM.add!(mpm, LLVM.AlwaysInlinerPass())
    vendored_buildEarlyOptimizerPipeline(mpm, job, opt_level)
    LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
        # Doesn't call instcombine
        GPUCompiler.buildLoopOptimizerPipeline(fpm, job, opt_level)
        vendored_buildScalarOptimizerPipeline(fpm, job, opt_level)
        if GPUCompiler.uses_julia_runtime(job) && opt_level >= 2
            # XXX: we disable vectorization, as this generally isn't useful for GPU targets
            #      and actually causes issues with some back-end compilers (like Metal).
            # TODO: Make this not dependent on `uses_julia_runtime` (likely CPU), but it's own control
            # Doesn't call instcombine
            GPUCompiler.buildVectorPipeline(fpm, job, opt_level)
        end
        # if isdebug(:optim)
        #     add!(fpm, WarnMissedTransformationsPass())
        # end
    end
    vendored_buildIntrinsicLoweringPipeline(mpm, job, opt_level)
    return GPUCompiler.buildCleanupPipeline(mpm, job, opt_level)
end

# compile to executable machine code
function compile(job)
    # lower to PTX
    # TODO: on 1.9, this actually creates a context. cache those.
    entry = GPUCompiler.JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(
            # :llvm, job; optimize=false, cleanup=false, validate=false, libraries=true
            :llvm,
            job;
            # :llvm, job; optimize=false, cleanup=false, validate=true, libraries=false
            # :llvm, job; optimize=false, cleanup=false, validate=false, libraries=false
        )

        if !Reactant.precompiling()
            GPUCompiler.link_library!(mod, GPUCompiler.load_runtime(job))
        end
        entryname = LLVM.name(meta.entry)

        if Reactant.Compiler.DUMP_LLVMIR[]
            println("cuda.jl immediate IR\n", string(mod))
        end
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
                vendored_buildNewPMPipeline!(mpm, job, opt_level)
            end
            LLVM.run!(pb, mod, tm)
        end
        if Reactant.Compiler.DUMP_LLVMIR[]
            println("cuda.jl pre vendor IR\n", string(mod))
        end

        LLVM.@dispose pb = LLVM.NewPMPassBuilder() begin
            LLVM.add!(pb, LLVM.NewPMModulePassManager()) do mpm
                LLVM.add!(mpm, LLVM.AlwaysInlinerPass())
            end
            LLVM.run!(pb, mod, tm)
        end

        vendored_optimize_module!(job, mod)
        if Reactant.Compiler.DUMP_LLVMIR[]
            println("cuda.jl post vendor IR\n", string(mod))
        end
        LLVM.run!(GPUCompiler.DeadArgumentEliminationPass(), mod, tm)

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

        errors = GPUCompiler.check_ir!(job, GPUCompiler.IRError[], mod)
        unique!(errors)
        filter!(errors) do err
            (kind, bt, meta) = err
            if meta !== nothing
                if kind == GPUCompiler.UNKNOWN_FUNCTION && startswith(meta, "__nv")
                    return false
                end
            end
            return true
        end
        if Reactant.Compiler.DUMP_LLVMIR[]
            println("cuda.jl postopt IR\n", string(mod))
        end
        if !isempty(errors)
            throw(GPUCompiler.InvalidIRError(job, errors))
        end
        # LLVM.strip_debuginfo!(mod)
        modstr = string(mod)
        # This is a bit weird since we're taking a module from julia's llvm into reactant's llvm version
        # it is probably safer to reparse a string using the right llvm module api, so we will do that.
        mmod = MLIR.IR.Module(
            @ccall MLIR.API.mlir_c.ConvertLLVMStrToMLIR(
                modstr::Cstring, MLIR.IR.context()::MLIR.API.MlirContext
            )::MLIR.API.MlirModule
        )
        @assert mmod != C_NULL

        linkRes = @ccall MLIR.API.mlir_c.LinkInModule(
            MLIR.IR.mmodule()::MLIR.API.MlirModule,
            mmod::MLIR.API.MlirModule,
            entryname::Cstring,
        )::MLIR.API.MlirOperation

        String(Reactant.TracedUtils.get_attribute_by_name(linkRes, "sym_name"))
    end

    return LLVMFunc{job.source.specTypes.parameters[1],job.source.specTypes}(nothing, entry)
end

# link into an executable kernel
function link(job, compiled)
    # load as an executable kernel object
    return compiled
end

function abi_sizeof(@nospecialize(x))
    return sizeof(typeof(x))
end
function abi_sizeof(@nospecialize(x::CuTracedArray))
    return sizeof(Ptr)
end
function abi_sizeof(@nospecialize(x::CUDA.CuDeviceArray))
    return sizeof(Ptr)
end

function to_bytes(x)
    sz = abi_sizeof(x)
    ref = Ref(x)
    GC.@preserve ref begin
        ptr = Base.reinterpret(Ptr{UInt8}, Base.unsafe_convert(Ptr{Cvoid}, ref))
        vec = Vector{UInt8}(undef, sz)
        for i in 1:sz
            @inbounds vec[i] = Base.unsafe_load(ptr, i)
        end
        vec
    end
end

function Reactant.make_tracer(
    seen, @nospecialize(prev::CuTracedArray), @nospecialize(path), mode; kwargs...
)
    x = Base.unsafe_pointer_to_objref(Base.reinterpret(Ptr{Cvoid}, prev.ptr))
    x = x::TracedRArray
    Reactant.make_tracer(seen, x, path, mode; kwargs...)
    return prev
end

function Reactant.make_tracer(
    seen, @nospecialize(prev::CuTracedRNumber), @nospecialize(path), mode; kwargs...
)
    x = Base.unsafe_pointer_to_objref(Base.reinterpret(Ptr{Cvoid}, prev.ptr))
    x = x::TracedRNumber
    Reactant.make_tracer(seen, x, path, mode; kwargs...)
    return prev
end

function get_field_offset(T::Type, path)
    offset = 0
    current_type = T

    for field in path
        # Get the field index
        field_idx = if field isa Integer
            field
        else
            @assert field isa Symbol
            findfirst(==(field), fieldnames(current_type))
        end
        if field_idx === nothing
            error(
                "Field $field not found in type $current_type, fieldnames=$(fieldnames(current_type)) T=$T path=$path",
            )
        end

        # Add the offset of this field
        toffset = fieldoffset(current_type, field_idx)
        tcurrent_type = fieldtype(current_type, field_idx)
        offset += toffset

        # Update current_type to the field's type for next iteration
        current_type = tcurrent_type
    end

    return offset
end

Reactant.@reactant_overlay @noinline function (func::LLVMFunc{F,tt})(
    args...;
    convert=Val(true),
    blocks::CuDim=1,
    threads::CuDim=1,
    cooperative::Bool=false,
    shmem::Integer=0,
    call_kwargs...,
) where {F,tt}
    blockdim = CUDA.CuDim3(blocks)
    threaddim = CUDA.CuDim3(threads)
    mod = MLIR.IR.mmodule()

    if convert == Val(true)
        args = recudaconvert.(args)
    end

    mlir_args = MLIR.IR.Value[]
    restys = MLIR.IR.Type[]
    aliases = MLIR.IR.Attribute[]

    fname = func.entry

    wrapper_tys = MLIR.IR.Type[]
    ctx = MLIR.IR.context()
    cullvm_ty = MLIR.IR.Type(MLIR.API.mlirLLVMPointerTypeGet(ctx, 1))

    # linearize kernel arguments
    seen = Reactant.OrderedIdDict()
    kernelargsym = gensym("kernelarg")

    for (i, prev) in enumerate(Any[func.f, args...])
        Reactant.make_tracer(seen, prev, (kernelargsym, i), Reactant.NoStopTracedTrack)
    end
    wrapper_tys = MLIR.IR.Type[]
    for arg in values(seen)
        if !(arg isa TracedRArray || arg isa TracedRNumber)
            continue
        end
        push!(wrapper_tys, cullvm_ty)
    end

    sym_name = String(gensym("call_$fname"))
    CConv = MLIR.IR.Attribute(
        MLIR.API.mlirLLVMCConvAttrGet(ctx, MLIR.API.MlirLLVMCConvPTX_Kernel)
    )
    voidty = MLIR.IR.Type(MLIR.API.mlirLLVMVoidTypeGet(ctx))
    wrapftype = MLIR.IR.Type(
        MLIR.API.mlirLLVMFunctionTypeGet(voidty, length(wrapper_tys), wrapper_tys, false)
    )
    wrapfunc = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.llvm.func(;
            sym_name,
            sym_visibility=MLIR.IR.Attribute("private"),
            function_type=wrapftype,
            body=MLIR.IR.Region(),
            CConv,
        )
    end
    wrapbody = MLIR.IR.Block(wrapper_tys, [MLIR.IR.Location() for _ in wrapper_tys])
    push!(MLIR.IR.region(wrapfunc, 1), wrapbody)
    for i in 1:length(wrapper_tys)
        @ccall MLIR.API.mlir_c.ReactantFuncSetArgAttr(
            wrapfunc::MLIR.API.MlirOperation,
            (i - 1)::Csize_t,
            "llvm.noalias"::MLIR.API.MlirStringRef,
            MLIR.IR.UnitAttribute()::MLIR.API.MlirAttribute,
        )::Cvoid
    end

    wrapargs = MLIR.IR.Value[]
    argidx = 1

    symtab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
    gpufunc = MLIR.IR.lookup(symtab, fname)
    MLIR.IR.attr!(
        gpufunc,
        "CConv",
        MLIR.IR.Attribute(MLIR.API.mlirLLVMCConvAttrGet(ctx, MLIR.API.MlirLLVMCConvC)),
    )
    gpu_function_type = MLIR.IR.Type(
        Reactant.TracedUtils.get_attribute_by_name(gpufunc, "function_type")
    )

    trueidx = 1
    allocs = Union{Tuple{MLIR.IR.Value,MLIR.IR.Type},Nothing}[]

    llvmptr = MLIR.IR.Type(MLIR.API.mlirLLVMPointerTypeGet(ctx, 0))
    i8 = MLIR.IR.Type(UInt8)
    allargs = Any[func.f, args...]
    for a in allargs
        if sizeof(a) == 0
            push!(allocs, nothing)
            continue
        end

        # TODO check for only integer and explicitly non cutraced types
        MLIR.IR.block!(wrapbody) do
            argty = MLIR.IR.Type(
                MLIR.API.mlirLLVMFunctionTypeGetInput(gpu_function_type, trueidx - 1)
            )
            trueidx += 1
            c1 = MLIR.IR.result(
                MLIR.Dialects.llvm.mlir_constant(;
                    res=MLIR.IR.Type(Int64), value=MLIR.IR.Attribute(1)
                ),
                1,
            )
            alloc = MLIR.IR.result(
                MLIR.Dialects.llvm.alloca(
                    c1; elem_type=MLIR.IR.Attribute(argty), res=llvmptr
                ),
                1,
            )
            push!(allocs, (alloc, argty))

            sz = abi_sizeof(a)
            array_ty = MLIR.IR.Type(MLIR.API.mlirLLVMArrayTypeGet(MLIR.IR.Type(Int8), sz))
            cdata = MLIR.IR.result(
                MLIR.Dialects.llvm.mlir_constant(;
                    res=array_ty, value=MLIR.IR.DenseElementsAttribute(to_bytes(a))
                ),
                1,
            )
            MLIR.Dialects.llvm.store(cdata, alloc)
        end
    end

    argidx = 1
    for arg in values(seen)
        if !(arg isa TracedRArray || arg isa TracedRNumber)
            continue
        end

        paths = Reactant.TracedUtils.get_paths(arg)

        arg = arg.mlir_data
        arg = Reactant.TracedUtils.transpose_val(arg)
        push!(restys, MLIR.IR.type(arg))
        push!(mlir_args, arg)

        ctx = MLIR.IR.context()
        out_tup = Ref{Int64}(argidx - 1)
        push!(
            aliases,
            MLIR.IR.Attribute(
                GC.@preserve ctx out_tup MLIR.API.stablehloOutputOperandAliasGet(
                    ctx,
                    length(wrapper_tys) == 1 ? 0 : 1,
                    pointer_from_objref(out_tup),
                    argidx - 1,
                    0,
                    C_NULL,
                )
            ),
        )

        for p in paths
            if p[1] !== kernelargsym
                continue
            end
            # Get the allocation corresponding to which arg we're doing
            alloc = allocs[p[2]][1]

            # we need to now compute the offset in bytes of the path
            julia_arg = allargs[p[2]]

            offset = get_field_offset(typeof(julia_arg), p[3:end])
            MLIR.IR.block!(wrapbody) do
                ptr = MLIR.IR.result(
                    MLIR.Dialects.llvm.getelementptr(
                        alloc,
                        MLIR.IR.Value[];
                        res=llvmptr,
                        elem_type=i8,
                        rawConstantIndices=MLIR.IR.Attribute([Int32(offset)]),
                    ),
                    1,
                )
                MLIR.Dialects.llvm.store(MLIR.IR.argument(wrapbody, argidx), ptr)
            end
        end
        argidx += 1
    end

    MLIR.IR.block!(wrapbody) do
        for arg in allocs
            if arg === nothing
                continue
            end
            alloc, argty = arg
            argres = MLIR.IR.result(MLIR.Dialects.llvm.load(alloc; res=argty), 1)
            push!(wrapargs, argres)
        end
        MLIR.Dialects.llvm.call(
            wrapargs,
            MLIR.IR.Value[];
            callee=MLIR.IR.FlatSymbolRefAttribute(Base.String(fname)),
            op_bundle_sizes=MLIR.IR.Attribute(Int32[]),
        )
        MLIR.Dialects.llvm.return_(nothing)
    end

    output_operand_aliases = MLIR.IR.Attribute(aliases)

    blk_operands = MLIR.IR.Value[]
    for idx in
        (blockdim.x, blockdim.y, blockdim.z, threaddim.x, threaddim.y, threaddim.z, shmem)
        push!(blk_operands, Reactant.promote_to(Reactant.TracedRNumber{Int}, idx).mlir_data)
    end

    @assert length(restys) == length(aliases)
    call = MLIR.Dialects.enzymexla.kernel_call(
        blk_operands...;
        inputs=mlir_args,
        result_0=restys,
        fn=MLIR.IR.FlatSymbolRefAttribute(sym_name),
        output_operand_aliases=MLIR.IR.Attribute(output_operand_aliases),
        xla_side_effect_free=MLIR.IR.UnitAttribute(),
    )

    argidx = 1
    for arg in values(seen)
        if !(arg isa TracedRArray || arg isa TracedRNumber)
            continue
        end
        arg.mlir_data = Reactant.TracedUtils.transpose_val(MLIR.IR.result(call, argidx))
        argidx += 1
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
        config = GPUCompiler.CompilerConfig(
            CUDA.PTXCompilerTarget(; cap=llvm_cap, ptx=llvm_ptx, debuginfo),
            CUDA.CUDACompilerParams(; cap=cuda_cap, ptx=cuda_ptx);
            kernel,
            name,
            always_inline,
            optimize=false,
            cleanup=false,
            validate=false,
            libraries=false,
        )
        GPUCompiler.cached_compilation(cache, source, config, compile, link)
    end
    return Core.Typeof(res)(f, res.entry)
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(A::Type{<:CuTracedArray}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return A
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(A::Type{<:CuTracedRNumber}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return A
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(A::Type{<:CUDA.CuArray}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    T = eltype(A)
    N = ndims(A)
    if mode == Reactant.ArrayToConcrete && T <: Reactant.ReactantPrimitive
        if runtime isa Val{:PJRT}
            return Reactant.ConcretePJRTArray{T,N,Reactant.Sharding.ndevices(sharding)}
        elseif runtime isa Val{:IFRT}
            return Reactant.ConcreteIFRTArray{T,N}
        end
        error("Unsupported runtime $runtime")
    else
        TT = Reactant.traced_type_inner(T, seen, mode, track_numbers, sharding, runtime)
        TT === T && return A
        return Array{
            Reactant.traced_type_inner(
                T, seen, mode, track_numbers, Base.getproperty(sharding, 1), runtime
            ),
            N,
        }
    end
end

function Reactant.make_tracer(
    seen,
    @nospecialize(prev::CUDA.CuArray),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Reactant.Sharding.NoSharding()),
    @nospecialize(runtime),
    kwargs...,
)
    RT = Core.Typeof(prev)
    # XXX: If someone wants to shard the same array with different shardings, we need to
    #      somehow handle this correctly... Right now we just use the first sharding.
    if haskey(seen, prev)
        return seen[prev]
    end
    if mode == Reactant.ArrayToConcrete && eltype(RT) <: Reactant.ReactantPrimitive
        if runtime isa Val{:PJRT}
            return seen[prev] = Reactant.ConcretePJRTArray(Array(prev); sharding)
        elseif runtime isa Val{:IFRT}
            return seen[prev] = Reactant.ConcreteIFRTArray(Array(prev); sharding)
        end
        error("Unsupported runtime $runtime")
    end
    TT = Reactant.traced_type(eltype(RT), Val(mode), track_numbers, sharding, runtime)
    if TT === eltype(RT)
        return prev
    end
    newa = Array{TT,ndims(RT)}(undef, size(prev))
    seen[prev] = newa
    same = true
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            nv = Reactant.make_tracer(
                seen,
                pv,
                append_path(path, I),
                mode;
                track_numbers,
                sharding=Base.getproperty(sharding, I),
                runtime,
                kwargs...,
            )
            if pv !== nv
                same = false
            end
            @inbounds newa[I] = nv
        end
    end
    if same
        seen[prev] = prev
        return prev
    end
    return newa
end

# In Julia v1.11.3 precompiling this module caches bad code:
# <https://github.com/EnzymeAD/Reactant.jl/issues/614>.
@static if !Sys.isapple()
    @setup_workload begin
        Reactant.initialize_dialect()

        if Reactant.XLA.REACTANT_XLA_RUNTIME == "PJRT"
            client = Reactant.XLA.PJRT.CPUClient(; checkcount=false)
        elseif Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
            client = Reactant.XLA.IFRT.CPUClient(; checkcount=false)
        else
            error("Unsupported runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)")
        end

        @compile_workload begin
            @static if Reactant.precompilation_supported() && VERSION != v"1.11.3"
                function square_kernel!(x)
                    i = CUDA.threadIdx().x
                    x[i] *= x[i]
                    return nothing
                end

                function square!(x)
                    CUDA.@cuda blocks = 1 threads = length(x) square_kernel!(x)
                    return nothing
                end
                y = Reactant.ConcreteRArray([2.0]; client)
                Reactant.Compiler.compile_mlir(square!, (y,); optimize=false)

                if y isa Reactant.ConcreteIFRTArray
                    Reactant.XLA.free_buffer(y.data.buffer)
                    y.data.buffer.buffer = C_NULL
                else
                    for dat in y.data
                        Reactant.XLA.free_buffer(dat.buffer)
                        dat.buffer.buffer = C_NULL
                    end
                end
            end
        end

        Reactant.XLA.free_client(client)
        client.client = C_NULL
        Reactant.deinitialize_dialect()
        Reactant.clear_oc_cache()
    end
end

end # module ReactantCUDAExt
