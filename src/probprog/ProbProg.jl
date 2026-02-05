module ProbProg

using ..Reactant:
    MLIR, TracedUtils, AbstractRNG, TracedRArray, TracedRNumber, ConcreteRNumber
using ..Compiler: @compile

include("Types.jl")
include("Distributions.jl")
include("FFI.jl")
include("Modeling.jl")
include("Display.jl")
include("Stats.jl")
include("MH.jl")
include("MCMC.jl")

# Types.
export Trace, Constraint, Selection, Address, TraceEntry, TracedTrace

# Distributions.
export Distribution, Normal, Exponential, LogNormal

# Utility functions.
export get_choices, select, unflatten_trace, filter_entries_by_selection

# MCMC Statistics.
export mcmc_summary

# Core MLIR ops.
export sample, untraced_call, simulate, generate, mh, mcmc

# Gen-like helper functions.
export simulate_, generate_

# Debug utilities.
export clear_dump_buffer!, show_dumps

end
