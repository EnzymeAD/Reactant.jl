enum_basetype(::Type{<:Base.Enum{T}}) where {T} = T

abstract type AbstractReactantEnum{E<:Base.Enum} end

"""
    TracedEnum{E <: Base.Enum}

An enum of type `E` represented by a traced integer during compilation. Supports enum
comparisons, selection, integer conversion, and scalar broadcasting. Compiled results
are reconstructed as [`ConcreteEnum`](@ref) values.
"""
mutable struct TracedEnum{E<:Base.Enum} <: AbstractReactantEnum{E}
    value::TracedRNumber
end

"""
    ConcreteEnum{E <: Base.Enum, N <: AbstractConcreteNumber}

An enum of type `E` backed by a concrete runtime integer of type `N`. Created by
`to_rarray(enum; track_numbers=Number)` and returned by compiled enum computations.
Supports enum comparisons, integer conversion, conversion back to `E`, hashing, and
scalar broadcasting. Assigning a plain enum to a field of this type creates a concrete
integer of the same runtime type; it does not require an active compilation.
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

Base.Integer(x::AbstractReactantEnum) = _payload_integer(_enum_payload(x))
(::Type{T})(x::AbstractReactantEnum) where {T<:Integer} = _payload_to(T, _enum_payload(x))

# Unlike the host constructor, this does not check that the integer is a valid member.
function (::Type{E})(x::TracedRNumber{<:Integer}) where {E<:Base.Enum}
    return TracedEnum{E}(promote_to(TracedRNumber{enum_basetype(E)}, x))
end

function Base.convert(::Type{E}, x::ConcreteEnum{E}) where {E<:Base.Enum}
    return E(Integer(x))
end
(::Type{E})(x::ConcreteEnum{E}) where {E<:Base.Enum} = convert(E, x)

# Concrete enum wrappers have the same equality and hash as the plain enum.
Base.hash(x::ConcreteEnum{E}, h::UInt) where {E} = hash(E(x), h)
Base.hash(x::TracedEnum, h::UInt) = hash(_enum_payload(x), h)

function Base.convert(::Type{TracedEnum{E}}, x::E) where {E<:Base.Enum}
    return ReactantCore.promote_to_traced(x)
end

function Base.convert(::Type{ConcreteEnum{E,N}}, x::E) where {E<:Base.Enum,N}
    return ConcreteEnum{E,N}(convert(N, Integer(x)))
end

function Base.convert(::Type{ConcreteEnum{E}}, x::E) where {E<:Base.Enum}
    return to_rarray(x; track_numbers=Number)
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
    I = enum_basetype(E)
    return TracedEnum{E}(ifelse(pred, _traced_payload(I, x), _traced_payload(I, y)))
end
