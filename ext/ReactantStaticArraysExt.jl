module ReactantStaticArraysExt

using LinearAlgebra: LinearAlgebra, DimensionMismatch
using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map, overloaded_mapreduce
import Reactant.TracedLinearAlgebra: overloaded_mul, overloaded_dot, overloaded_norm

using StaticArrays: SArray, StaticArray, StaticArrayStyle
using Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle

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

# A static array broadcast with a traced number is unrolled, as with any other
# scalar.
function Base.BroadcastStyle(s::StaticArrayStyle, ::AbstractReactantArrayStyle{0})
    return s
end

# We don't want to overload map on StaticArrays since it is likely better to just unroll things
overloaded_map(f, a::StaticArray, rest::StaticArray...) = f.(a, rest...)
overloaded_mapreduce(f, op, a::SAReact; kwargs...) = mapreduce(f, op, a, kwargs...)

function overloaded_dot(x::StaticArray, y::StaticArray)
    if length(x) != length(y)
        throw(
            DimensionMismatch(
                lazy"x has length $(length(x)), but y has length $(length(y))"
            ),
        )
    end
    tx = Tuple(x)
    ty = Tuple(y)
    isempty(tx) && return LinearAlgebra.dot(zero(eltype(x)), zero(eltype(y)))
    acc = LinearAlgebra.dot(tx[1], ty[1])
    for i in 2:length(tx)
        acc += LinearAlgebra.dot(tx[i], ty[i])
    end
    return acc
end

function overloaded_norm(x::StaticArray, p::Real=2)
    tx = Tuple(x)
    if isempty(tx)
        T0 = float(real(Reactant.unwrapped_eltype(x)))
        return zero(T0)
    end
    p == 2 && return sqrt(real(overloaded_dot(x, x)))
    if isinf(p)
        acc = abs(tx[1])
        for i in 2:length(tx)
            acc = p > 0 ? max(acc, abs(tx[i])) : min(acc, abs(tx[i]))
        end
        return acc
    end
    if p == 1
        acc = abs(tx[1])
        for i in 2:length(tx)
            acc += abs(tx[i])
        end
        return acc
    end
    T = float(real(Reactant.unwrapped_eltype(x)))
    acc = abs(tx[1])^p
    for i in 2:length(tx)
        acc += abs(tx[i])^p
    end
    return acc^(T(1 / p))
end

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
