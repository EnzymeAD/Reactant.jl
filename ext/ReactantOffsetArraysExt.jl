module ReactantOffsetArraysExt

using OffsetArrays
using OffsetArrays: OffsetArray
using Reactant: Reactant, MLIR, Ops, TracedRArray

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{<:OffsetArray}), @nospecialize(args...)
)
    N = ndims(OA)
    T = OffsetArrays.parenttype(OA)
    T2 = Reactant.traced_type_inner(T, args...)
    return OffsetArray{eltype(T2),N,T2}
end

end
