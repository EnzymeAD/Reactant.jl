module Reactant

using Adapt: Adapt, WrappedArray

# auxiliary types and functions
include("OrderedIdDict.jl")

using Enzyme

abstract type RArray{T,N} <: AbstractArray{T,N} end
abstract type RNumber{T} <: Number end

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
include("TracedRArray.jl")
include("Tracing.jl")
include("Compiler.jl")

using .Compiler: @compile, @code_hlo, traced_getfield, create_result, compile
export ConcreteRArray, @compile, @code_hlo

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
    backend = XLA.backends[backend]
    return XLA.default_backend[] = backend
end

end # module
