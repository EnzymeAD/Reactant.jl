# `Base.Enum` values (`@enum`, EnumX's `@enumx`, ...) are isbits reinterpretations of an
# integer. A traced enum is an enum-typed wrapper around the traced base integer — not a
# `TracedRNumber` with an enum element type — so the payload is an ordinary traced number
# and the wrapper rides the generic struct tracing and result reconstruction. The wrapper
# is mutable with an untyped slot because the result write-back machinery replaces traced
# payloads with concrete numbers via `setfield!`.

enum_basetype(::Type{<:Base.Enum{T}}) where {T} = T

"""
    TracedEnum{E <: Base.Enum}

A value of the enum type `E` carried through a Reactant compilation. `value` holds the
enum's base integer: a `TracedRNumber` while tracing, a concrete number in the result of a
compiled call. Supports the enum operations (`==`, `!=`, ordered comparisons, `ifelse`)
against other `TracedEnum`s and plain `E` values, `Integer`/integer-type conversion, and —
once the payload is concrete — conversion back to `E`.
"""
mutable struct TracedEnum{E<:Base.Enum}
    value
end

function ReactantCore.promote_to_traced(x::E) where {E<:Base.Enum}
    return TracedEnum{E}(promote_to(TracedRNumber{enum_basetype(E)}, Integer(x)))
end

_enum_payload(x::TracedEnum) = getfield(x, :value)
_enum_payload(x::Base.Enum) = Integer(x)

_payload_integer(v::AbstractConcreteNumber) = Integer(to_number(v))
_payload_integer(v) = v

function _payload_to(::Type{T}, v::TracedRNumber) where {T<:Integer}
    return promote_to(TracedRNumber{T}, v)
end
_payload_to(::Type{T}, v::AbstractConcreteNumber) where {T<:Integer} = T(to_number(v))
_payload_to(::Type{T}, v) where {T<:Integer} = T(v)

function _traced_payload(::Type{I}, x::Base.Enum) where {I}
    return promote_to(TracedRNumber{I}, Integer(x))
end
_traced_payload(::Type{I}, x::TracedEnum) where {I} = _enum_payload(x)

# Conversions

Base.Integer(x::TracedEnum) = _payload_integer(_enum_payload(x))
(::Type{T})(x::TracedEnum) where {T<:Integer} = _payload_to(T, _enum_payload(x))

# Unlike the host constructor, this does not check that the integer is a valid member.
function (::Type{E})(x::TracedRNumber{<:Integer}) where {E<:Base.Enum}
    return TracedEnum{E}(promote_to(TracedRNumber{enum_basetype(E)}, x))
end

function Base.convert(::Type{E}, x::TracedEnum{E}) where {E<:Base.Enum}
    v = _enum_payload(x)
    v isa TracedRNumber &&
        error("cannot convert a traced $E back to the plain enum during tracing")
    return E(Integer(_payload_integer(v)))
end
(::Type{E})(x::TracedEnum{E}) where {E<:Base.Enum} = convert(E, x)

# With a concrete payload the wrapper is value-equal to `E`, so it hashes like `E` too.
function Base.hash(x::TracedEnum{E}, h::UInt) where {E<:Base.Enum}
    v = _enum_payload(x)
    v isa TracedRNumber && return hash(v, h)
    return hash(E(Integer(_payload_integer(v))), h)
end

function Base.convert(::Type{TracedEnum{E}}, x::E) where {E<:Base.Enum}
    return ReactantCore.promote_to_traced(x)
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
        function $(jlop)(lhs::TracedEnum{E}, rhs::TracedEnum{E}) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
        function $(jlop)(lhs::TracedEnum{E}, rhs::E) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
        function $(jlop)(lhs::E, rhs::TracedEnum{E}) where {E}
            return $(jlop)(_enum_payload(lhs), _enum_payload(rhs))
        end
    end
end

function Base.ifelse(
    pred::TracedRNumber{Bool}, x::Union{TracedEnum{E},E}, y::Union{TracedEnum{E},E}
) where {E<:Base.Enum}
    I = enum_basetype(E)
    return TracedEnum{E}(ifelse(pred, _traced_payload(I, x), _traced_payload(I, y)))
end

# Tracing. The wrapper itself is an ordinary struct handled by the generic machinery; only
# the plain `Base.Enum` value needs entry points, and they place the payload at the
# wrapper-relative path (`value` is field 1).

Base.@nospecializeinfer function should_track_enum(
    @nospecialize(E::Type{<:Base.Enum}), @nospecialize(track_numbers::Type)
)
    return E <: track_numbers || enum_basetype(E) <: track_numbers
end

Base.@nospecializeinfer function traced_type_inner(
    @nospecialize(T::Type{<:Base.Enum}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    should_track_enum(T, track_numbers) || return T
    if mode == ArrayToConcrete ||
        mode == NoStopTracedTrack ||
        mode == TracedTrack ||
        mode == TracedSetPath
        return TracedEnum{T}
    end
    return T
end

Base.@nospecializeinfer function make_tracer(
    seen,
    @nospecialize(prev::Base.Enum),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    @nospecialize(device = nothing),
    @nospecialize(client = nothing),
    kwargs...,
)
    if mode == TracedToTypes
        push!(path, prev)
        return nothing
    end
    RT = Core.Typeof(prev)
    should_track_enum(RT, track_numbers) || return prev
    if mode == ArrayToConcrete
        runtime isa Val{:PJRT} && return TracedEnum{RT}(
            ConcretePJRTNumber(Integer(prev); sharding, device, client)
        )
        runtime isa Val{:IFRT} && return TracedEnum{RT}(
            ConcreteIFRTNumber(Integer(prev); sharding, device, client)
        )
        error("Unsupported runtime $runtime")
    elseif mode == NoStopTracedTrack
        payload = TracedRNumber{enum_basetype(RT)}(
            (append_path(path, 1),), @opcall(constant(Integer(prev))).mlir_data
        )
        seen[gensym("enum")] = payload
        return TracedEnum{RT}(payload)
    elseif mode == TracedToConcrete
        throw("Input is not a traced-type: $(RT)")
    end
    return prev
end

@inline function to_rarray_internal(
    @nospecialize(x::Base.Enum),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    runtime,
    @nospecialize(device),
    @nospecialize(client)
)
    should_track_enum(typeof(x), track_numbers) || return x
    if runtime isa Val{:PJRT}
        return TracedEnum{typeof(x)}(
            ConcretePJRTNumber(Integer(x); sharding, device, client)
        )
    elseif runtime isa Val{:IFRT}
        return TracedEnum{typeof(x)}(
            ConcreteIFRTNumber(Integer(x); sharding, device, client)
        )
    end
    return error("Unsupported runtime $runtime")
end
