module MLIR

using ..Reactant

module API
    using CEnum: @cenum
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll
    using Libdl: Libdl

    # MLIR C API
    let
        include("libMLIR_h.jl")
    end

    # Additional functions
    function EnzymeJaXMapSymbol(name, sym)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Ptr{Cvoid}
        )::Cvoid
    end
    function EnzymeJaXMapSymbol(name, sym::Integer)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Int64
        )::Cvoid
    end

    function RegisterEnzymeXLACPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLACPUHandler()::Cvoid
    end

    function RegisterEnzymeXLAGPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLAGPUHandler()::Cvoid
    end

    function registerEnzymeJaXXLAFFI()
        if Libdl.dlsym(
            Reactant_jll.libReactantExtra_handle,
            :registerEnzymeJaXXLAFFI;
            throw_error=false,
        ) === nothing
            @debug "registerEnzymeJaXXLAFFI not found in libReactantExtra, skipping \
                    registration of EnzymeJaXXLAFFI. Update Reactant_jll to resolve this."
            return nothing
        end
        @ccall Reactant_jll.libReactantExtra.registerEnzymeJaXXLAFFI()::Cvoid
    end
end # module API

include("IR/IR.jl")
include("Dialects.jl")
include("Highlight.jl")

end # module MLIR
