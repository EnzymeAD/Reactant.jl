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

export ProbProgTrace, Constraint, Selection, CompiledFnCache, Address
export get_choices, select, choicemap, with_compiled_cache

export sample, call, simulate, generate

end
