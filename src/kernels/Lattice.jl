"""
    Lattice

A module for defining blocked-programs (kernels) for heterogeneous computing. Currently we
only lower to NVIDIA GPUs via Triton, but eventually we will support other backends.
"""
module Lattice

using ..Reactant:
    Reactant, MLIR, Ops, TracedUtils, TracedRArray, TracedRNumber, TTPtr, unwrapped_eltype

include("api.jl")
include("call.jl")

end
