module ReactantStaticArraysExt

using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map
import Reactant.TracedLinearAlgebra: overloaded_mul

using StaticArrays: SArray, StaticArray

const SAReact{Sz, T} = StaticArray{Sz, T} where {Sz <: Tuple, T <: Reactant.TracedRNumber}

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

function Reactant.materialize_traced_array(x::SAReact)
    return x
end

# We don't want to overload map on StaticArrays because it autopromote to TracedRArrays which we 
# do not want.
overloaded_map(f, a::SAReact) = f.(a)
overloaded_mul(A::SAReact, B::SAReact) = A * B

end
