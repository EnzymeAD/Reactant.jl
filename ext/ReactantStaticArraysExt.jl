module ReactantStaticArraysExt

using Reactant
import Reactant.TracedRArrayOverrides: overloaded_map, overloaded_mapreduce
import Reactant.TracedLinearAlgebra: overloaded_mul, overloaded_dot
using LinearAlgebra: dot
using StaticArrays: SArray, StaticArray

const SARArray{Sz,T,N} = StaticArray{Sz,T,N} where {Sz<:Tuple,T<:Reactant.TracedRNumber}
const SARVector{N, T, N} = SARArray{Tuple{N},T,N}
const SARMatrix{M,N,T} = SARArray{Tuple{M,N},T,N}

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

function Reactant.materialize_traced_array(x::SARArray)
    return x
end

# We don't want to overload map on StaticArrays since it is likely better to just unroll things
overloaded_map(f, a::SARArray, rest::SARArray...) = f.(a, rest...)
overloaded_mapreduce(f, op, a::SARArray; kwargs...) = mapreduce(f, op, a, kwargs...)

function overloaded_mul(A::SARArray, B::SARArray, alpha::Number=true, beta::Number=false)
    # beta is not supported since it is zero by default in Reactant 
    # (similar is zero'd automatically for TracedRArrays)
    C = A * B
    if !(alpha isa Reactant.TracedRNumber) && isone(alpha)
        return C
    end
    return C .* alpha
end

function overloaded_dot(x::SARArray, y::SARArray)
    return dot(x, y)
end

function overloaded_dot(x::SARVector, y::SARMatrix, z::SARVector)
    return dot(x, y, z)
end

end
