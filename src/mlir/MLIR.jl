module MLIR

module API
    using CEnum
    using Preferences
    using Reactant_jll

    const mlir_c = Reactant_jll.libReactantExtra

    # MLIR C API
    let
        include("libMLIR_h.jl")
    end
end # module API

include("IR/IR.jl")

include("Dialects.jl")

const registry = Ref{MLIR.IR.DialectRegistry}()

function __init__()
    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

end # module MLIR
