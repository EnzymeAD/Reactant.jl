module MLIR

module API
using CEnum
using JLLWrappers
using Preferences
using Libdl

const mlir_c = joinpath(@__DIR__, "../../deps/ReactantExtra/bazel-bin/libReactantExtra.so")

# MLIR C API
let
    include(joinpath(@__DIR__, "../../deps/ReactantExtra/bazel-bin/libMLIR_h.jl"))
end
end # module API

include("IR/IR.jl")

include("Dialects.jl")


end # module MLIR
