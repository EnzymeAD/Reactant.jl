module ReactantStaticArraysExt

using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map, overloaded_mapreduce
import Reactant.TracedLinearAlgebra: overloaded_mul, overloaded_dot, overloaded_norm

using StaticArrays: SArray, StaticArray, StaticVector

const SAReact{Sz,T} = StaticArray{Sz,T} where {Sz<:Tuple,T<:Reactant.TracedRNumber}

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

# We don't want to overload map on StaticArrays since it is likely better to just unroll things.
# This is also needed when only some of the inputs hold traced numbers (e.g. `muladd` of a
# plain `SVector` and an `SVector` of traced numbers is implemented with `map`).
overloaded_map(f, a::StaticArray, rest::StaticArray...) = f.(a, rest...)
overloaded_mapreduce(f, op, a::SAReact; kwargs...) = mapreduce(f, op, a, kwargs...)

# A static vector of traced scalars is a tuple of scalars: do not send it through XLA.
function overloaded_dot(x::StaticVector, y::StaticVector)
    length(x) == length(y) || throw(
        DimensionMismatch(lazy"x has length $(length(x)), but y has length $(length(y))")
    )
    return sum(conj.(x) .* y)
end

function _static_norm(x::StaticArray, p::Real)
    p == 2 && return sqrt(sum(abs2, x))
    p == 1 && return sum(abs, x)
    p == Inf && return maximum(abs, x)
    p == -Inf && return minimum(abs, x)
    return sum(a -> abs(a)^p, x)^(1 / p)
end
overloaded_norm(x::StaticArray, p::Real=2) = _static_norm(x, p)
overloaded_norm(x::StaticVector, p::Real=2) = _static_norm(x, p)

function overloaded_mul(A::SAReact, B::SAReact, alpha::Number=true, beta::Number=false)
    # beta is not supported since it is zero by default in Reactant 
    # (similar is zero'd automatically for TracedRArrays)
    C = A * B
    if !(alpha isa Reactant.TracedRNumber) && isone(alpha)
        return C
    end
    return C .* alpha
end

end
