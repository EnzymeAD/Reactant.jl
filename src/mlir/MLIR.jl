module MLIR

using ..Reactant

module API
    using CEnum: @cenum
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll

    const mlir_c = if Reactant_jll.is_available()
        Reactant_jll.libReactantExtra
    else
        missing
    end

    # MLIR C API
    let
        include("libMLIR_h.jl")
    end

    # Additional functions
    function EnzymeJaXMapSymbol(name, sym)
        @ccall mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end

    function RegisterEnzymeXLACPUHandler()
        @ccall mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
    end

    function RegisterEnzymeXLAGPUHandler()
        @ccall mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid
    end
end # module API

include("IR/IR.jl")

include("Dialects.jl")

end # module MLIR
