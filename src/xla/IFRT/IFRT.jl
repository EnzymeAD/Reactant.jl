module IFRT

using ..Reactant: Reactant, MLIR
using ..XLA: XLA

include("Client.jl")
include("Device.jl")
include("Sharding.jl")

end
