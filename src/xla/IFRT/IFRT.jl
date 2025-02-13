module IFRT

using CEnum

import ..XLA
import .XLA: hold!
import ..MLIR

include("LoadedExecutable.jl")
include("Client.jl")
include("Array.jl")

end
