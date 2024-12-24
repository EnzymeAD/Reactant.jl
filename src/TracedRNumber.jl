module TracedRNumberOverrides

using ..Reactant:
    Reactant, TracedRNumber, TracedRArray, ReactantPrimitive, TracedUtils, Ops, MLIR
using ReactantCore

ReactantCore.is_traced(::TracedRNumber) = true

Base.getindex(a::TracedRNumber{T}) where {T} = a

Base.zero(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

function Base.eps(::Type{TracedRNumber{T}}) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, eps(T))
end

function Base.show(io::IOty, X::TracedRNumber{T}) where {T,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRNumber{", T, "}(", X.paths, ")")
end

Base.only(A::TracedRNumber{T}) where {T} = A

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

# Bool has special promotion rules in Base
function Base.promote_rule(::Type{Bool}, ::Type{TracedRNumber{T}}) where {T}
    return TracedRNumber{T}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Bool}) where {T}
    return TracedRNumber{T}
end

function Base.promote_rule(::Type{T}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

# NOTE: This is inconsistent with the behavior of `convert` but we do it since it is a very
#       common usecase
TracedRNumber{T}(x::TracedRNumber{T}) where {T} = x
TracedRNumber{T}(x::TracedRNumber) where {T} = TracedUtils.promote_to(TracedRNumber{T}, x)
TracedRNumber{T}(x::Number) where {T} = TracedUtils.promote_to(TracedRNumber{T}, x)

function TracedUtils.promote_to(::Type{TracedRNumber{T}}, rhs) where {T}
    if rhs isa TracedRNumber
        rhs isa TracedRNumber{T} && return rhs
        return Ops.convert(TracedRNumber{T}, rhs)
    end
    if rhs isa TracedRArray{<:Any,0}
        return TracedUtils.promote_to(
            TracedRNumber{T},
            TracedRNumber{Reactant.unwrapped_eltype(rhs)}((), rhs.mlir_data),
        )
    end
    rhs isa Number &&
        return TracedUtils.promote_to(TracedRNumber{T}, Ops.constant(fill(T(rhs))))
    return TracedUtils.promote_to(TracedRNumber{T}, Ops.constant(collect(rhs)))
end

function TracedUtils.promote_to(::TracedRNumber{T}, rhs) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, rhs)
end

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
    (:(Base.:*), :multiply),
    (:(Base.:/), :divide),
    (:(Base.:^), :power),
)
    @eval function $(jlop)(
        @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
    ) where {T}
        return Ops.$(hloop)(lhs, rhs)
    end
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), rhs, ::typeof(RoundDown)
) where {T<:Integer}
    return Ops.divide(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end

for (jlop, hloop, hlocomp) in (
    (:(Base.:(==)), :compare, "EQ"),
    (:(Base.:(!=)), :compare, "NE"),
    (:(Base.:(>=)), :compare, "GE"),
    (:(Base.:(>)), :compare, "GT"),
    (:(Base.:(<=)), :compare, "LE"),
    (:(Base.:(<)), :compare, "LT"),
)
    @eval begin
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return Ops.compare(lhs, rhs; comparison_direction=$(hlocomp))
        end

        function $(jlop)(@nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs)) where {T}
            return $(jlop)(lhs, TracedUtils.promote_to(lhs, rhs))
        end
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(jlop)(lhs, TracedUtils.promote_to(lhs, rhs))
        end

        function $(jlop)(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber{T})) where {T}
            return $(jlop)(TracedUtils.promote_to(rhs, lhs), rhs)
        end
        function $(jlop)(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(jlop)(TracedUtils.promote_to(rhs, lhs), rhs)
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
        ) where {T1,T2}
            commonTy = TracedRNumber{Base.promote_type(T1, T2)}
            lhs = TracedUtils.promote_to(commonTy, lhs)
            rhs = TracedUtils.promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T1}),
    @nospecialize(y::TracedRNumber{T2})
) where {T1,T2}
    @warn "`ifelse` with different element-types in Reactant works by promoting the \
            element-type to the common type. This is semantically different from the \
            behavior of `ifelse` in Base. Use with caution" maxlog = 1
    T = promote_type(T1, T2)
    return ifelse(
        pred,
        TracedUtils.promote_to(TracedRNumber{T}, x),
        TracedUtils.promote_to(TracedRNumber{T}, y),
    )
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T})
) where {T}
    return Ops.select(pred, x, y)
end

for (T1, T2) in zip((Bool, Integer), (Bool, Integer))
    T = promote_type(T1, T2)
    @eval begin
        function Base.:&(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.and(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        function Base.:|(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.or(
                TracedUtils.promote_to(TracedRNumber{$(T)}, x),
                TracedUtils.promote_to(TracedRNumber{$(T)}, y),
            )
        end
        Base.:!(x::TracedRNumber{<:$(T1)}) = Ops.not(x)
    end
end

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRNumber{T}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
    (:(Base.abs), :abs),
    (:(Base.:-), :negate),
    (:(Base.sin), :sine),
    (:(Base.cos), :cosine),
    (:(Base.tanh), :tanh),
    (:(Base.FastMath.tanh_fast), :tanh),
    (:(Base.exp), :exponential),
    (:(Base.FastMath.exp_fast), :exponential),
    (:(Base.expm1), :exponential_minus_one),
    (:(Base.log), :log),
    (:(Base.log1p), :log_plus_one),
    (:(Base.sqrt), :sqrt),
    (:(Base.ceil), :ceil),
    (:(Base.floor), :floor),
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber)) = Ops.$(hloop)(lhs)
end

Base.conj(x::TracedRNumber) = x
Base.conj(x::TracedRNumber{<:Complex}) = Ops.conj(x)

Base.real(x::TracedRNumber) = x
Base.real(x::TracedRNumber{<:Complex}) = Ops.real(x)

Base.imag(x::TracedRNumber) = zero(x)
Base.imag(x::TracedRNumber{<:Complex}) = Ops.imag(x)

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval Base.clamp(x::TracedRNumber, min::$(minT), max::$(maxT)) = Ops.clamp(min, x, max)
end

function Base.fill(x::TracedRNumber, dims::NTuple{N,Integer}) where {N}
    return TracedUtils.broadcast_to_size(x, dims)
end

function Base.float(x::TracedRNumber{T}) where {T}
    return TracedUtils.promote_to(TracedRNumber{float(T)}, x)
end

# Concatenation. Numbers in Julia are handled in a much less generic fashion than arrays
Base.vcat(x::TracedRNumber...) = Base.typed_vcat(Base.promote_eltypeof(x...), x...)
function Base.typed_vcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_vcat(T, map(Base.Fix2(TracedUtils.broadcast_to_size, (1,)), x)...)
end

Base.hcat(x::TracedRNumber...) = Base.typed_hcat(Base.promote_eltypeof(x...), x...)
function Base.typed_hcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_hcat(T, map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), x)...)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, xs::TracedRNumber...)
    return Base.typed_hvcat(Base.promote_eltypeof(xs...), rows, xs...)
end
function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Vararg{Int}}, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvcat(T, rows, xs...)
end

function Base.hvncat(dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...)
    return Base.typed_hvncat(Base.promote_eltypeof(xs...), dims, row_first, xs...)
end
function Base.typed_hvncat(
    ::Type{T}, dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(TracedUtils.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvncat(T, dims, row_first, xs...)
end

end
