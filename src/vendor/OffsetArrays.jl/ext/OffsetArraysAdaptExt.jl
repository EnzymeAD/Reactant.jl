module OffsetArraysAdaptExt

using OffsetArrays, Adapt

##
# Adapt allows for automatic conversion of CPU OffsetArrays to GPU OffsetArrays
##
import Adapt
Adapt.adapt_structure(to, O::OffsetArray) = OffsetArrays.parent_call(x -> Adapt.adapt(to, x), O)

@static if isdefined(Adapt, :parent_type)
    # To support Adapt 3.0 which doesn't have parent_type defined
    Adapt.parent_type(::Type{OffsetArray{T,N,AA}}) where {T,N,AA} = AA
end

end
