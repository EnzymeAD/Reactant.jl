module ReactantOffsetArraysExt

using OffsetArrays: OffsetArray
using Reactant: Reactant, MLIR, Ops, TracedRArray

function Reactant.traced_type(
    ::Type{<:OffsetArray{<:Any,N,T}}, seen::ST, ::Val{mode}, track_numbers
) where {T,N,ST,mode}
    T2 = Reactant.traced_type(T, seen, Val(mode), track_numbers)
    return OffsetArray{eltype(T2),N,T2}
end

end
