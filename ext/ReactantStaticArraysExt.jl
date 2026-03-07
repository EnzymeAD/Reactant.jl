module ReactantStaticArraysExt

using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map

using StaticArrays: SArray, StaticArray

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(FA::Type{SArray{S,T,N,L}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {S,T,N,L}
    T_traced = Reactant.traced_type_inner(T, seen, mode, track_numbers, ndevices, runtime)
    return SArray{S,T_traced,N,L}
end

function Reactant.materialize_traced_array(x::SArray)
    as = Reactant.aos_to_soa(collect(x))
    return Reactant.materialize_traced_array(as)
end

# We don't want to overload map on StaticArrays because it autopromote to TracedRArrays which we 
# do not want.
overloaded_map(f, a::StaticArray{<:Tuple,<:Reactant.TracedRNumber}) = f.(a)

end
