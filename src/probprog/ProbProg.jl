module ProbProg

using ..Reactant:
    MLIR,
    TracedUtils,
    AbstractConcreteArray,
    AbstractConcreteNumber,
    AbstractRNG,
    TracedRArray,
    TracedRNumber,
    ConcreteRNumber,
    Ops
using ..Compiler: @jit, @compile
using Enzyme

include("Types.jl")
include("FFI.jl")
include("Modeling.jl")
include("Display.jl")

# Types.
export ProbProgTrace, Constraint, Selection, Address

# Utility functions.
export get_choices, select

# Core MLIR ops.
export sample, untraced_call, simulate, generate

# Gen-like helper functions.
export simulate_, generate_

end
