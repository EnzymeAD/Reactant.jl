module ReactantStaticArraysExt

using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map, overloaded_mapreduce
import Reactant.TracedLinearAlgebra: overloaded_mul

using StaticArrays: SArray, StaticArray

const SAReact{Sz, T} = StaticArray{Sz, T} where {Sz<:Tuple, T<:Reactant.TracedRNumber}

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

# We don't want to overload map on StaticArrays since it is likely better to just unroll things
overloaded_map(f, a::SAReact, rest::SAReact...) = f.(a, rest...)
overloaded_mapreduce(f, op, a::SAReact; kwargs...) = mapreduce(f, op, a, kwargs...)

function overloaded_mul(
    A::SAReact, B::SAReact, alpha::Number=true, beta::Number=false
)
    C = A * B
    if !(alpha isa Reactant.TracedRNumber) && isone(alpha)
        return C
    end
    return C .* alpha
end

end
