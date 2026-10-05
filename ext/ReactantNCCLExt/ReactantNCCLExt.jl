module ReactantNCCLExt

using Reactant
using Reactant: @reactant_overlay, TracedRArray, TracedRNumber
using NCCL

mutable struct TracedCommunicator
    handle
end

Reactant.Ops.mlir_type(::Type{TracedCommunicator}) = error("to do")

include("Ops.jl")

end
