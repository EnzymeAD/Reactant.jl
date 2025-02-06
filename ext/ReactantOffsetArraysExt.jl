module ReactantOffsetArraysExt

using OffsetArrays
using OffsetArrays: OffsetArray
using Reactant: Reactant, MLIR, Ops, TracedRArray

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{<:OffsetArray}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type = Union{})
)
    N = ndims(OA)
    T = OffsetArrays.parenttype(OA)
    T2 = Reactant.traced_type_inner(T, seen, mode, track_numbers)
    return OffsetArray{eltype(T2),N,T2}
end

end
