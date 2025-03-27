module TracedROffsetArrayOverrides

#------------------------------------------------------------------------------
# Vendored from: https://github.com/JuliaArrays/OffsetArrays.jl/tree/master
#------------------------------------------------------------------------------

include("./vendor/OffsetArrays.jl/src/OffsetArrays.jl")

const TracedROffsetArray = OffsetArrays.OffsetArray

#------------------------------------------------------------------------------

using ReactantCore
using Adapt
using Base: @propagate_inbounds


ReactantCore.is_traced(::TracedROffsetArray, seen) = true
ReactantCore.is_traced(::TracedROffsetArray) = true

function Adapt.parent_type(::Type{TracedROffsetArray{T,N,AA}}) where {T,N,AA}
    return TracedROffsetArray{T,N,AA}
end

end
