module CAPI

using Reactant_jll: libReactantExtra as libxla

"""
    csizeof(::Type{T}) where {T}

Return the size in bytes occupied by the XLA C data structure.
"""
function csizeof end

include("PJRT.jl")
include("FFI.jl")
include("TPU.jl")

end
