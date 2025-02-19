module IFRT

using ..Reactant: Reactant, MLIR
using ..XLA: XLA

include("Client.jl")
include("Device.jl")
include("Memory.jl")
include("Future.jl")
include("Sharding.jl")
include("Array.jl")
include("AsyncArray.jl")
include("LoadedExecutable.jl")

end
