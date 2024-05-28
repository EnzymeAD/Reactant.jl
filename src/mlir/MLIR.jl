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

end # module MLIR
