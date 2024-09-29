module Reactant

# auxiliary types and functions
include("OrderedIdDict.jl")

include("mlir/MLIR.jl")
include("XLA.jl")
include("Interpreter.jl")

abstract type RArray{T,N} <: AbstractArray{T,N} end

function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end

include("utils.jl")
include("ConcreteRArray.jl")
include("TracedRArray.jl")

include("Tracing.jl")
include("Compiler.jl")

using .Compiler: @compile, @code_hlo, traced_getfield, create_result, compile
export ConcreteRArray, @compile, @code_hlo
using .XLA: set_default_backend

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

end # module
