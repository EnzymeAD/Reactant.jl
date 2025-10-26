module ProbProg

using ..Reactant:
    MLIR, TracedUtils, AbstractRNG, TracedRArray, TracedRNumber, ConcreteRNumber
using ..Compiler: @jit, @compile

include("Types.jl")
include("FFI.jl")
include("Modeling.jl")
include("Display.jl")
include("MH.jl")
include("HMC.jl")

# Types.
export ProbProgTrace, Constraint, Selection, Address

# Utility functions.
export get_choices, select
export to_trace_tensor, from_trace_tensor
export to_constraint_tensor, from_constraint_tensor

# Core MLIR ops.
export sample, untraced_call, simulate, generate, mh, hmc

# Gen-like helper functions.
export simulate_, generate_

end
