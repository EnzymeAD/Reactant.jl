module Accelerators

using ..Reactant: Reactant

include("Registration.jl")

include("CPU.jl")
include("GPU.jl") # TODO: disambiguate between CUDA/ROCM
include("TPU.jl")
include("Metal.jl")
include("TT.jl")

end
