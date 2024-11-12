module Reactant

using ReactantCore: ReactantCore, @trace, MissingTracedValue

using LinearAlgebra: LinearAlgebra
using Adapt: Adapt, WrappedArray
using GPUArraysCore: GPUArraysCore

# auxiliary types and functions
include("OrderedIdDict.jl")

using Enzyme

@static if isdefined(Core, :BFloat16)
    const ReactantPrimitive = Union{
        Bool,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float16,
        Core.BFloat16,
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    }
else
    const ReactantPrimitive = Union{
        Bool,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float16,
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    }
end

abstract type RArray{T<:ReactantPrimitive,N} <: AbstractArray{T,N} end
abstract type RNumber{T<:ReactantPrimitive} <: Number end

function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:RArray}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if RT <: ConcreteRArray
        res = RT(zeros(eltype(RT), size(prev)))
        seen[prev] = res
        return res
    end

    if RT <: TracedRArray
        res = broadcast_to_size(eltype(RT)(0), size(prev))
        seen[prev] = res
        return res
    end

    attr = fill(MLIR.IR.Attribute(eltype(RT)(0)), mlir_type(prev))
    cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    res = RT((), cst)
    seen[prev] = res
    return res
end

include("mlir/MLIR.jl")
include("XLA.jl")
include("Interpreter.jl")

include("utils.jl")

include("ConcreteRArray.jl")
include("TracedRNumber.jl")
include("TracedRArray.jl")

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("ControlFlow.jl")
include("Tracing.jl")
include("Compiler.jl")

using .Compiler: @compile, @code_hlo, @jit, traced_getfield, create_result, compile
export ConcreteRArray, ConcreteRNumber, @compile, @code_hlo, @jit, @trace

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

function set_default_backend(backend::XLA.Client)
    if backend === XLA.backends["cpu"]
        setting = GPUArraysCore.ScalarAllowed
    else
        setting = GPUArraysCore.default_scalar_indexing()
    end
    task_local_storage(:ScalarIndexing, setting)
    GPUArraysCore.requested_scalar_indexing[] = setting

    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    return set_default_backend(XLA.backends[backend])
end

end # module
