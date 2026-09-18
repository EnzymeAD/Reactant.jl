enum_basetype(::Type{<:Base.Enum{T}}) where {T} = T

abstract type AbstractReactantEnum{E<:Base.Enum} end

"""
    TracedEnum{E <: Base.Enum, I <: Integer}

An enum of type `E` backed by a traced integer of base type `I`. Supports comparisons,
selection, integer conversion, and scalar broadcasting. Compiled results use
[`ConcreteEnum`](@ref).
"""
mutable struct TracedEnum{E<:Base.Enum,I<:Integer} <: AbstractReactantEnum{E}
    value::TracedRNumber{I}
end

TracedEnum{E}(value::TracedRNumber{I}) where {E,I} = TracedEnum{E,I}(value)

"""
    ConcreteEnum{E <: Base.Enum, N <: AbstractConcreteNumber}

An enum of type `E` backed by a concrete runtime integer of type `N`. Created by
`to_rarray(enum; track_numbers=Base.Enum)` and returned by compiled enum computations.
Supports enum comparisons, integer conversion, conversion back to `E`, hashing, and
scalar broadcasting. Assigning a plain enum to a field of this type creates a concrete
integer, as for `ConcreteRNumber` fields.
"""
mutable struct ConcreteEnum{E<:Base.Enum,N<:AbstractConcreteNumber} <:
               AbstractReactantEnum{E}
    value::N
end

ConcreteEnum{E}(value::N) where {E,N<:AbstractConcreteNumber} = ConcreteEnum{E,N}(value)

Base.broadcastable(x::AbstractReactantEnum) = Ref(x)

function ReactantCore.promote_to_traced(x::E) where {E<:Base.Enum}
    return TracedEnum{E}(promote_to(TracedRNumber{enum_basetype(E)}, Integer(x)))
end

_enum_payload(x::AbstractReactantEnum) = getfield(x, :value)
_enum_payload(x::Base.Enum) = Integer(x)

# Conversions

Base.Integer(x::TracedEnum) = x.value
Base.Integer(x::ConcreteEnum) = Integer(to_number(x.value))
(::Type{T})(x::TracedEnum) where {T<:Integer} = promote_to(TracedRNumber{T}, x.value)
(::Type{T})(x::ConcreteEnum) where {T<:Integer} = T(to_number(x.value))

# Enum membership is not checked during tracing.
function (::Type{E})(x::TracedRNumber{<:Integer}) where {E<:Base.Enum}
    return TracedEnum{E}(promote_to(TracedRNumber{enum_basetype(E)}, x))
end

function Base.convert(::Type{E}, x::ConcreteEnum{E}) where {E<:Base.Enum}
    return E(Integer(x))
end
(::Type{E})(x::ConcreteEnum{E}) where {E<:Base.Enum} = convert(E, x)

# Preserve enum hashes while allowing payloads outside the enum's declared values.
function Base.hash(x::ConcreteEnum{E}, h::UInt) where {E}
    v = Integer(x)
    haskey(Base.Enums.namemap(E), v) && return hash(E(v), h)
    return hash(v, hash(E, h))
end

function Base.convert(::Type{T}, x::E) where {E<:Base.Enum,T<:TracedEnum{E}}
    return convert(T, ReactantCore.promote_to_traced(x))
end

function Base.convert(::Type{ConcreteEnum{E,N}}, x::E) where {E<:Base.Enum,N}
    return ConcreteEnum{E,N}(convert(N, Integer(x)))
end

function Base.convert(::Type{ConcreteEnum{E}}, x::E) where {E<:Base.Enum}
    return to_rarray(x; track_numbers=Base.Enum)
end

# Comparisons and selection

for jlop in (
    :(Base.:(==)),
    :(Base.:(!=)),
    :(Base.:(>=)),
    :(Base.:(>)),
    :(Base.:(<=)),
    :(Base.:(<)),
    :(Base.isless),
)
    @eval begin
        function $(jlop)(
            lhs::AbstractReactantEnum{E}, rhs::AbstractReactantEnum{E}
        ) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
        function $(jlop)(lhs::AbstractReactantEnum{E}, rhs::E) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
        function $(jlop)(lhs::E, rhs::AbstractReactantEnum{E}) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
    end
end

function Base.ifelse(
    pred::TracedRNumber{Bool}, x::Union{TracedEnum{E},E}, y::Union{TracedEnum{E},E}
) where {E<:Base.Enum}
    return TracedEnum{E}(ifelse(pred, _enum_payload(x), _enum_payload(y)))
end

# Enums are traced leaves; their paths refer to the wrapper.

TracedUtils.get_mlir_data(x::TracedEnum) = TracedUtils.get_mlir_data(x.value)
function TracedUtils.set_mlir_data!(x::TracedEnum, data)
    TracedUtils.set_mlir_data!(x.value, data)
    return x
end
TracedUtils.get_paths(x::TracedEnum) = TracedUtils.get_paths(x.value)
function TracedUtils.set_paths!(x::TracedEnum, paths)
    TracedUtils.set_paths!(x.value, paths)
    return x
end

# Type mappings

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:ConcreteEnum}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    # `T` may be the UnionAll `ConcreteEnum{E}`.
    E = Base.unwrap_unionall(T).parameters[1]
    E isa TypeVar && return T
    if mode == ConcreteToTraced
        return TracedEnum{E,enum_basetype(E)}
    elseif mode == ArrayToConcrete
        T isa UnionAll && return T
        N = traced_type_inner(T.parameters[2], seen, mode, track_numbers, ndevices, runtime)
        return ConcreteEnum{E,N}
    end
    return T
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:TracedEnum}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    E = Base.unwrap_unionall(T).parameters[1]
    E isa TypeVar && return T
    if mode == TracedToConcrete
        N = traced_type_inner(
            TracedRNumber{enum_basetype(E)}, seen, mode, track_numbers, ndevices, runtime
        )
        return ConcreteEnum{E,N}
    end
    mode == ConcreteToTraced && throw("Cannot trace existing trace type")
    return T
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:Base.Enum}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    T <: track_numbers || return T
    if mode == ArrayToConcrete
        N = traced_type_inner(enum_basetype(T), seen, mode, Number, ndevices, runtime)
        return ConcreteEnum{T,N}
    elseif mode == NoStopTracedTrack || mode == TracedTrack || mode == TracedSetPath
        return TracedEnum{T,enum_basetype(T)}
    end
    return T
end

# Tracers

# Keep the payload at the enum's path so plain and wrapped enums share branch paths.
Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Base.Enum),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    kwargs...,
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    E = Core.Typeof(prev)
    E <: track_numbers || return prev
    payload = make_tracer(seen, Integer(prev), path, mode; track_numbers=Number, kwargs...)
    payload isa TracedRNumber && return TracedEnum{E}(payload)
    payload isa AbstractConcreteNumber && return ConcreteEnum{E}(payload)
    return prev
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::ConcreteEnum{E}), @nospecialize(path), mode; kwargs...
) where {E}
    mode == TracedToTypes && throw("Cannot have ConcreteEnum as function call argument.")
    if mode == ArrayToConcrete
        return ConcreteEnum{E}(make_tracer(seen, prev.value, path, mode; kwargs...))
    end
    mode != ConcreteToTraced && throw("Cannot trace existing trace type")
    haskey(seen, prev) && return seen[prev]
    res = TracedEnum{E}(make_tracer(seen, prev.value, path, mode; kwargs...))
    seen[prev] = res
    return res
end

Base.@nospecializeinfer function make_tracer(
    seen, @nospecialize(prev::TracedEnum{E}), @nospecialize(path), mode; kwargs...
) where {E}
    mode == ConcreteToTraced && throw("Cannot trace existing trace type")
    mode == TracedToTypes && return make_tracer_unknown(seen, prev, path, mode; kwargs...)
    if mode != NoStopTracedTrack && haskey(seen, prev)
        return seen[prev]
    end
    payload = make_tracer(seen, prev.value, path, mode; kwargs...)
    res = if payload === prev.value
        prev
    elseif payload isa TracedRNumber
        TracedEnum{E}(payload)
    elseif payload isa AbstractConcreteNumber
        ConcreteEnum{E}(payload)
    else
        payload
    end
    seen[prev] = res
    return res
end
