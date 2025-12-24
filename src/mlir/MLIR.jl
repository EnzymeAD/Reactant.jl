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

    # MLIR C API - extra
    function mlirComplexAttrDoubleGet(ctx, type, real, imag)
        @ccall mlir_c.mlirComplexAttrDoubleGet(
            ctx::MlirContext, type::MlirType, real::Cdouble, imag::Cdouble
        )::MlirAttribute
    end

    function mlirComplexAttrDoubleGetChecked(loc, type, real, imag)
        @ccall mlir_c.mlirComplexAttrDoubleGetChecked(
            loc::MlirLocation, type::MlirType, real::Cdouble, imag::Cdouble
        )::MlirAttribute
    end
end # module API

include("IR/IR.jl")

include("Dialects.jl")

end # module MLIR
