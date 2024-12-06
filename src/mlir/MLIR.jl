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
