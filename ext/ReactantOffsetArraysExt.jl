module ReactantOffsetArraysExt

using OffsetArrays
using OffsetArrays: OffsetArray
using Reactant: Reactant, MLIR, Ops, TracedRArray

Base.@nospecializeinfer function Reactant.traced_type(
        @nospecialize(OA::Type{<:OffsetArray}), seen::ST, ::Val{mode}, track_numbers
) where {ST,mode}
    N = ndims(OA)
    T = OffsetArrays.parenttype(OA)
    T2 = Reactant.traced_type(T, seen, Val(mode), track_numbers)
    return OffsetArray{eltype(T2),N,T2}
end

end
