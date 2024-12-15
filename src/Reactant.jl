module Reactant

using ReactantCore: ReactantCore, @trace, MissingTracedValue

using LinearAlgebra: LinearAlgebra
using Adapt: Adapt, WrappedArray
using GPUArraysCore: GPUArraysCore, @allowscalar, allowscalar # keep this import to allow users to do `Reactant.allowscalar(false)`

export @allowscalar # re-exported from GPUArraysCore

# auxiliary types and functions
include("OrderedIdDict.jl")

using Enzyme

struct ReactantABI <: Enzyme.EnzymeCore.ABI end

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

Base.collect(A::RArray) = copy(A)

function ancestor(x::AbstractArray)
    p_x = parent(x)
    p_x === x && return x
    return ancestor(p_x)
end

include("mlir/MLIR.jl")
include("XLA.jl")
include("Interpreter.jl")

include("utils.jl")

mutable struct TracedRArray{T,N} <: RArray{T,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        shape = Tuple(shape)
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

const WrappedTracedRArray{T,N} = WrappedArray{T,N,TracedRArray,TracedRArray{T,N}}
const AnyTracedRArray{T,N} = Union{TracedRArray{T,N},WrappedTracedRArray{T,N}}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = Union{
    AnyTracedRArray{T,2},LinearAlgebra.Diagonal{T,TracedRArray{T,1}}
}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

function TracedRArray(data::MLIR.IR.Value)
    data_type = MLIR.IR.type(data)
    return TracedRArray{eltype(MLIR.IR.julia_type(data_type)),ndims(data_type)}(
        (), data, size(data_type)
    )
end

mutable struct TracedRNumber{T} <: RNumber{T}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}

    function TracedRNumber{T}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
    ) where {T}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == ()
        end
        return new{T}(paths, mlir_data)
    end
end

struct XLAArray{T,N} <: RArray{T,N}
    # size::NTuple{N,Int}
end

mutable struct ConcreteRArray{T,N} <: RArray{T,N}
    data::XLA.AsyncBuffer
    #   data::XLAArray{T, N}
    shape::NTuple{N,Int}
end

const WrappedConcreteRArray{T,N} = WrappedArray{T,N,ConcreteRArray,ConcreteRArray{T,N}}
const AnyConcreteRArray{T,N} = Union{ConcreteRArray{T,N},WrappedConcreteRArray{T,N}}

mutable struct ConcreteRNumber{T} <: RNumber{T}
    data::XLA.AsyncBuffer
end

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

include("Ops.jl")
include("TracedUtils.jl")

include("TracedRNumber.jl")
include("TracedRArray.jl")

include("ConcreteRArray.jl")

include("linear_algebra.jl")

include("ControlFlow.jl")
include("Tracing.jl")
include("Compiler.jl")

include("Overlay.jl")

function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:RArray}
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

using .Compiler: @compile, @code_hlo, @jit, create_result, compile
export ConcreteRArray, ConcreteRNumber, @compile, @code_hlo, @jit, @trace

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

function set_default_backend(backend::XLA.Client)
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    return set_default_backend(XLA.backends[backend])
end

end # module
