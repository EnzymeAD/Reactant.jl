module Reactant

using ReactantCore: ReactantCore, @trace, MissingTracedValue

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using Functors: @leaf

using Adapt: Adapt, WrappedArray
using GPUArraysCore: GPUArraysCore, @allowscalar, allowscalar # keep this import to allow users to do `Reactant.allowscalar(false)`

export @allowscalar # re-exported from GPUArraysCore

# auxiliary types and functions
include("OrderedIdDict.jl")

using Enzyme

struct ReactantABI <: Enzyme.EnzymeCore.ABI end

include("PrimitiveTypes.jl")

function ancestor(x::AbstractArray)
    p_x = parent(x)
    p_x === x && return x
    return ancestor(p_x)
end

function ancestor(T::Type{<:AbstractArray})
    if applicable(Adapt.parent_type, T)
        p_T = Adapt.parent_type(T)
        p_T == T && return T
        return ancestor(p_T)
    end
    @warn "`Adapt.parent_type` is not implemented for $(T). Assuming $T isn't a wrapped \
           array." maxlog = 1
    return T
end

include("mlir/MLIR.jl")
include("XLA.jl")
include("Sharding.jl")
include("Devices.jl")
include("Interpreter.jl")
include("Profiler.jl")
include("Types.jl")

const with_profiler = Profiler.with_profiler

export Sharding

include("utils.jl")

function TracedRArray{T}(data::MLIR.IR.Value) where {T}
    data_type = MLIR.IR.type(data)
    if T == eltype(MLIR.IR.julia_type(data_type))
        return TracedRArray{T,ndims(data_type)}((), data, size(data_type))
    end
    tdata = TracedRArray(data)
    return Ops.convert(TracedRArray{T,ndims(data_type)}, tdata)
end

function TracedRArray(data::MLIR.IR.Value)
    return TracedRArray{eltype(MLIR.IR.julia_type(MLIR.IR.type(data)))}(data)
end

unwrapped_eltype(::Type{T}) where {T<:Number} = T
unwrapped_eltype(::Type{<:RNumber{T}}) where {T} = T
unwrapped_eltype(::Type{TracedRNumber{T}}) where {T} = T

unwrapped_eltype(::T) where {T<:Number} = T
unwrapped_eltype(::RNumber{T}) where {T} = T
unwrapped_eltype(::TracedRNumber{T}) where {T} = T

unwrapped_eltype(::Type{<:RArray{T,N}}) where {T,N} = T
unwrapped_eltype(::Type{<:AbstractArray{T,N}}) where {T,N} = unwrapped_eltype(T)
unwrapped_eltype(::Type{<:AnyTracedRArray{T,N}}) where {T,N} = T

unwrapped_eltype(::RArray{T,N}) where {T,N} = T
unwrapped_eltype(::AbstractArray{T,N}) where {T,N} = unwrapped_eltype(T)
unwrapped_eltype(::AnyTracedRArray{T,N}) where {T,N} = T

aos_to_soa(x::AbstractArray) = x
aos_to_soa(x::AnyTracedRArray) = x
function aos_to_soa(x::AbstractArray{ConcreteRNumber{T}}) where {T}
    x_c = ConcreteRArray(zeros(T, size(x)))
    x_c .= x
    return x_c
end
function aos_to_soa(x::AbstractArray{TracedRNumber{T}}) where {T}
    for i in eachindex(x)
        if !isassigned(x, i)
            x[i] = TracedUtils.promote_to(TracedRNumber{T}, 0)
        end
    end
    return Ops.reshape(vcat(x...), size(x)...)
end

include("Ops.jl")
include("TracedUtils.jl")

include("TracedRNumber.jl")
include("TracedRArray.jl")

include("ConcreteRArray.jl")

use_overlayed_version(iter) = any(use_overlayed_version, iter)

use_overlayed_version(::TracedRArray) = true
use_overlayed_version(::TracedRNumber) = true
use_overlayed_version(::Number) = false
use_overlayed_version(::MissingTracedValue) = true
use_overlayed_version(::TracedRNG) = true

function use_overlayed_version(x::AbstractArray)
    a = ancestor(x)
    a === x && return false
    return use_overlayed_version(a)
end

# StdLib Overloads
include("stdlibs/LinearAlgebra.jl")
include("stdlibs/Random.jl")
include("stdlibs/Base.jl")

# Other Integrations
include("Enzyme.jl")

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("ControlFlow.jl")
include("Tracing.jl")
include("Compiler.jl")

include("Overlay.jl")

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:Union{RArray,RNumber}}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(eltype(RT), nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    res = zero(prev)
    seen[prev] = res
    return res
end

using .Compiler: @compile, @code_hlo, @jit, traced_getfield, create_result, compile
export ConcreteRArray, ConcreteRNumber, @compile, @code_hlo, @jit, @trace

const registry = Ref{Union{Nothing,MLIR.IR.DialectRegistry}}()

function initialize_dialect()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

function deinitialize_dialect()
    return registry[] = nothing
end

using Libdl
using Reactant_jll
using LLVMOpenMP_jll
function initialize_ptrs()
    for name in (
        "__kmpc_barrier",
        "__kmpc_global_thread_num",
        "__kmpc_for_static_fini",
        "__kmpc_for_static_init_8u",
        "__kmpc_fork_call",
    )
        sym = Libdl.dlsym(LLVMOpenMP_jll.libomp_handle, name)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end
    # TODO on next jll bump (0.61) change this to call ReactantHermeticCudaGetVersion
    if (@ccall MLIR.API.mlir_c.ReactantCudaDriverGetVersion()::UInt32) != 0
        for name in (
            "cuLaunchKernel",
            "cuModuleLoadData",
            "cuModuleGetFunction",
            "cuStreamSynchronize",
        )
            sym = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, name)
            @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
        end
    end
end

function __init__()
    initialize_ptrs()
    return initialize_dialect()
end

function set_default_backend(backend::XLA.Client)
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    return set_default_backend(XLA.backends[backend])
end

include("Precompile.jl")

end # module
