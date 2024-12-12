module Reactant

using ReactantCore: ReactantCore, @trace, MissingTracedValue

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG

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

abstract type RNumber{T<:ReactantPrimitive} <: Number end

abstract type RArray{T,N} <: AbstractArray{T,N} end

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
include("Interpreter.jl")

include("utils.jl")

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

mutable struct TracedRArray{T,N} <: RArray{TracedRNumber{T},N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        shape = Tuple(shape)
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape "Expected: $(shape), got: $(size(MLIR.IR.type(mlir_data)))"
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

Adapt.parent_type(::Type{TracedRArray{T,N}}) where {T,N} = TracedRArray{T,N}

const WrappedTracedRArray{T,N} = WrappedArray{
    TracedRNumber{T},N,TracedRArray,TracedRArray{T,N}
}
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

struct XLAArray{T,N} <: RArray{T,N} end

Adapt.parent_type(::Type{XLAArray{T,N}}) where {T,N} = XLAArray{T,N}

mutable struct ConcreteRNumber{T} <: RNumber{T}
    data::XLA.AsyncBuffer
end

mutable struct ConcreteRArray{T,N} <: RArray{T,N}
    data::XLA.AsyncBuffer
    shape::NTuple{N,Int}
end

Adapt.parent_type(::Type{ConcreteRArray{T,N}}) where {T,N} = ConcreteRArray{T,N}

const WrappedConcreteRArray{T,N} = WrappedArray{T,N,ConcreteRArray,ConcreteRArray{T,N}}
const AnyConcreteRArray{T,N} = Union{ConcreteRArray{T,N},WrappedConcreteRArray{T,N}}

unwrapped_eltype(::Type{T}) where {T<:Number} = T
unwrapped_eltype(::Type{<:RNumber{T}}) where {T} = T
unwrapped_eltype(::Type{<:TracedRNumber{T}}) where {T} = T

unwrapped_eltype(::T) where {T<:Number} = T
unwrapped_eltype(::RNumber{T}) where {T} = T
unwrapped_eltype(::TracedRNumber{T}) where {T} = T

unwrapped_eltype(::Type{<:RArray{T,N}}) where {T,N} = T
unwrapped_eltype(::Type{<:AbstractArray{T,N}}) where {T,N} = unwrapped_eltype(T)
unwrapped_eltype(::Type{<:AnyTracedRArray{T,N}}) where {T,N} = T

unwrapped_eltype(::RArray{T,N}) where {T,N} = T
unwrapped_eltype(::AbstractArray{T,N}) where {T,N} = unwrapped_eltype(T)
unwrapped_eltype(::AnyTracedRArray{T,N}) where {T,N} = T

include("Ops.jl")
include("TracedUtils.jl")

include("TracedRNumber.jl")
include("TracedRArray.jl")

include("ConcreteRArray.jl")

mutable struct TracedRNG <: Random.AbstractRNG
    seed::Union{ConcreteRArray{UInt64,1},TracedRArray{UInt64,1}}
    const algorithm::String
end

# StdLib Overloads
include("stdlibs/LinearAlgebra.jl")
include("stdlibs/Random.jl")

const TracedType = Union{TracedRArray,TracedRNumber,MissingTracedValue}

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
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    return set_default_backend(XLA.backends[backend])
end

end # module
