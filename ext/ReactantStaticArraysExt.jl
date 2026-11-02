module ReactantStaticArraysExt

using GPUArraysCore: @allowscalar
using Reactant:
    Reactant,
    AnyTracedRArray,
    CallWithReactant,
    ReactantPrimitive,
    TracedLinearAlgebra,
    TracedRArrayOverrides,
    TracedRNumber
using Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle
using ReactantCore: ReactantCore, materialize_traced_array
using StaticArrays: SArray, StaticArray, StaticArrayStyle

const SAReact{Sz,T} = StaticArray{Sz,T} where {Sz<:Tuple,T<:TracedRNumber}

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

function ReactantCore.materialize_traced_array(x::SAReact)
    return x
end

# A static array broadcast with a traced number is unrolled, as with any other
# scalar.
function Base.BroadcastStyle(s::StaticArrayStyle, ::AbstractReactantArrayStyle{0})
    return s
end

# Convert non-static traced array arguments (e.g. `TracedRArray`) into a plain `Array` of
# `TracedRNumber`s so `StaticArrays.map` can index them without triggering disallowed
# scalar indexing, while keeping `!(arg isa StaticArray)` so `StaticArrays._map` still
# picks the first actual `StaticArray` for `similar_type`.
_unroll_arg(::Tuple, x::AbstractArray) = x
function _unroll_arg(sz::Tuple, x::AnyTracedRArray{T,N}) where {T,N}
    (x isa StaticArray || x isa Array) && return x
    size(x) == sz ||
        throw(DimensionMismatch("Sizes $((sz, size(x))) of input arrays do not match"))
    x = materialize_traced_array(x)
    return reshape(TracedRNumber{T}[@allowscalar(x[i]) for i in 1:prod(sz)], sz)
end

# We don't want to overload map on StaticArrays since it is likely better to just unroll
# things. The method signatures mirror the ones StaticArrays overloads for `Base.map` (a
# `StaticArray` in one of the first two positions). Deferring to StaticArrays' `map` keeps
# the static return type (`SVector`, `SMatrix`, `FieldVector`, ...) regardless of whether
# the traced values live in the static arrays, in the other arrays or in `f`. The generic
# `overloaded_map` can't handle these since a `StaticArray` of `TracedRNumber`s is not
# materializable into a `TracedRArray`.
function TracedRArrayOverrides.overloaded_map(f, a::StaticArray, rest::AbstractArray...)
    sz = size(a)
    return map(CallWithReactant(f), a, map(Base.Fix1(_unroll_arg, sz), rest)...)
end
function TracedRArrayOverrides.overloaded_map(
    f, a::AbstractArray, b::StaticArray, rest::AbstractArray...
)
    sz = size(b)
    return map(
        CallWithReactant(f), _unroll_arg(sz, a), b, map(Base.Fix1(_unroll_arg, sz), rest)...
    )
end
function TracedRArrayOverrides.overloaded_map(
    f, a::StaticArray, b::StaticArray, rest::AbstractArray...
)
    sz = size(a)
    return map(CallWithReactant(f), a, b, map(Base.Fix1(_unroll_arg, sz), rest)...)
end

for A in (
    StaticArray,
    StaticArray{<:Tuple,<:ReactantPrimitive},
    StaticArray{<:Tuple,TracedRNumber{T}} where {T},
)
    @eval function TracedRArrayOverrides.overloaded_mapreduce(f, op, a::$A; kwargs...)
        return mapreduce(CallWithReactant(f), op, a; kwargs...)
    end
end

function TracedLinearAlgebra.overloaded_mul(
    A::SAReact, B::SAReact, alpha::Number=true, beta::Number=false
)
    # beta is not supported since it is zero by default in Reactant
    # (similar is zero'd automatically for TracedRArrays)
    C = A * B
    if !(alpha isa TracedRNumber) && isone(alpha)
        return C
    end
    return C .* alpha
end

end
