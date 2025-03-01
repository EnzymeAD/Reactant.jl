module TracedRNumberOverrides

using ..Reactant:
    Reactant, TracedRNumber, TracedRArray, TracedUtils, Ops, MLIR, unwrapped_eltype
using ReactantCore

ReactantCore.is_traced(::TracedRNumber) = true

Base.getindex(a::TracedRNumber{T}) where {T} = a

Base.zero(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = TracedUtils.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

function Base.eps(::Type{TracedRNumber{T}}) where {T}
    return TracedUtils.promote_to(TracedRNumber{T}, eps(T))
end

function Base.isfinite(x::TracedRNumber{<:Complex})
    return isfinite(real(x)) & isfinite(imag(x))
end
Base.isfinite(x::TracedRNumber{<:AbstractFloat}) = Ops.is_finite(x)

function Base.isnan(x::TracedRNumber{<:Complex})
    return isnan(real(x)) | isnan(imag(x))
end
function Base.isnan(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return !isfinite(x) & (x != typemax(T)) & (x != typemin(T))
end

Base.isinf(x::TracedRNumber{<:Complex}) = isinf(real(x)) | isinf(imag(x))
Base.isinf(x::TracedRNumber{<:AbstractFloat}) = Ops.is_inf(x)
Base.isinf(::TracedRNumber{<:Integer}) = false

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

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{S}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

# NOTE: This is inconsistent with the behavior of `convert` but we do it since it is a very
#       common usecase
TracedRNumber{T}(x::TracedRNumber{T}) where {T} = x
function TracedRNumber{T}(x::TracedRNumber) where {T}
    return TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end
function TracedRNumber{T}(x::Number) where {T}
    return TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end

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
    rhs isa Number && return TracedUtils.promote_to(TracedRNumber{T}, Ops.fill(T(rhs)))
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
    (:(Base.rem), :remainder),
)
    @eval function $(jlop)(
        @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
    ) where {T}
        return Ops.$(hloop)(lhs, rhs)
    end
end

function Base.rem(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
) where {T}
    return Ops.remainder(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end
function Base.rem(
    @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
) where {T}
    return Ops.remainder(TracedUtils.promote_to(TracedRNumber{T}, lhs), rhs)
end

# Based on https://github.com/JuliaLang/julia/blob/39255d47db7657950ff1c82137ecec5a70bae622/base/float.jl#L608-L617
function Base.mod(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    r = rem(x, y)
    return ifelse(r == 0, copysign(r, y), ifelse((r > 0) ⊻ (y > 0), r + y, r))
end
function Base.mod(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
) where {T}
    return mod(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end
function Base.mod(
    @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
) where {T}
    return mod(TracedUtils.promote_to(TracedRNumber{T}, lhs), rhs)
end

function Base.div(@nospecialize(lhs::TracedRNumber{T}), rhs) where {T<:Integer}
    return Ops.divide(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), rhs, ::typeof(RoundDown)
) where {T<:Integer}
    return Ops.divide(lhs, TracedUtils.promote_to(TracedRNumber{T}, rhs))
end

function Base.:/(
    @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::TracedRNumber{T})
) where {T<:Integer}
    return float(lhs) / float(rhs)
end

for (jlop, hloop, hlocomp) in (
    (:(Base.:(==)), :compare, "EQ"),
    (:(Base.:(!=)), :compare, "NE"),
    (:(Base.:(>=)), :compare, "GE"),
    (:(Base.:(>)), :compare, "GT"),
    (:(Base.:(<=)), :compare, "LE"),
    (:(Base.:(<)), :compare, "LT"),
    (:(Base.isless), :compare, "LT"),
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

function Base.ifelse(@nospecialize(pred::TracedRNumber{Bool}), x::Number, y::Number)
    return ifelse(
        pred,
        TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(x)}, x),
        TracedUtils.promote_to(TracedRNumber{unwrapped_eltype(y)}, y),
    )
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
        function Base.xor(x::TracedRNumber{<:$(T1)}, y::TracedRNumber{<:$(T2)})
            return Ops.xor(
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
    (:(Base.tan), :tan),
    (:(Base.tanh), :tanh),
    (:(Base.FastMath.tanh_fast), :tanh),
    (:(Base.exp), :exponential),
    (:(Base.FastMath.exp_fast), :exponential),
    (:(Base.expm1), :exponential_minus_one),
    (:(Base.log), :log),
    (:(Base.log1p), :log_plus_one),
    (:(Base.sqrt), :sqrt),
    (:(Base.acos), :acos),
    (:(Base.acosh), :acosh),
    (:(Base.asin), :asin),
    (:(Base.asinh), :asinh),
    (:(Base.atan), :atan),
    (:(Base.atanh), :atanh),
    (:(Base.sign), :sign),
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber)) = Ops.$(hloop)(lhs)
end

for (jlop, hloop) in
    ((:(Base.sinpi), :sine), (:(Base.cospi), :cosine), (:(Base.tanpi), :tan))
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{T})) where {T} = Ops.$(hloop)(T(π) * lhs)
end

Base.sincospi(x::TracedRNumber{T}) where {T} = Ops.sine(T(π) * x), Ops.cosine(T(π) * x)

Base.conj(x::TracedRNumber) = x
Base.conj(x::TracedRNumber{<:Complex}) = Ops.conj(x)

Base.real(x::TracedRNumber) = x
Base.real(x::TracedRNumber{<:Complex}) = Ops.real(x)

Base.isreal(::TracedRNumber) = false
Base.isreal(::TracedRNumber{<:Real}) = true

Base.imag(x::TracedRNumber) = zero(x)
Base.imag(x::TracedRNumber{<:Complex}) = Ops.imag(x)

Base.iseven(x::TracedRNumber) = iseven(real(x))
function Base.iseven(x::TracedRNumber{<:Real})
    return iszero(
        rem(
            TracedUtils.promote_to(TracedRNumber{Int}, x),
            TracedUtils.promote_to(TracedRNumber{Int}, 2),
        ),
    )
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval Base.clamp(x::TracedRNumber, min::$(minT), max::$(maxT)) = Ops.clamp(min, x, max)
end

function Base.fill(x::TracedRNumber, dims::NTuple{N,Integer}) where {N}
    return TracedUtils.broadcast_to_size(x, dims)
end
function Base.fill(x::TracedRNumber, ::Tuple{})
    return TracedUtils.broadcast_to_size(x, ())
end

function Base.float(x::TracedRNumber{T}) where {T}
    return TracedUtils.promote_to(TracedRNumber{float(T)}, x)
end

using Reactant: ReactantFloat

Base.round(A::TracedRNumber{<:ReactantFloat}) = Ops.round_nearest_even(A)
Base.floor(A::TracedRNumber{<:ReactantFloat}) = Ops.floor(A)
Base.ceil(A::TracedRNumber{<:ReactantFloat}) = Ops.ceil(A)

function Base.unsafe_trunc(
    T::Type{<:Reactant.ReactantInt}, x::TracedRNumber{<:Reactant.ReactantFloat}
)
    return Ops.convert(TracedRNumber{T}, x)
end

for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for Tf in (Float16, Float32, Float64)
        if Ti <: Unsigned || sizeof(Ti) < sizeof(Tf)
            # Here `Tf(typemin(Ti))-1` is exact, so we can compare the lower-bound
            # directly. `Tf(typemax(Ti))+1` is either always exactly representable, or
            # rounded to `Inf` (e.g. when `Ti==UInt128 && Tf==Float32`).
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRNumber{$Tf})
                    # TODO throw error within traced
                    # if $(Tf(typemin(Ti))-one(Tf)) < x < $(Tf(typemax(Ti))+one(Tf))
                    return Base.unsafe_trunc($Ti, x)
                    # else
                    #     throw(Base.InexactError(:trunc, $Ti, x))
                    # end
                end
            end
        else
            # Here `eps(Tf(typemin(Ti))) > 1`, so the only value which can be truncated to
            # `Tf(typemin(Ti)` is itself. Similarly, `Tf(typemax(Ti))` is inexact and will
            # be rounded up. This assumes that `Tf(typemin(Ti)) > -Inf`, which is true for
            # these types, but not for `Float16` or larger integer types.
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRNumber{$Tf})
                    # TODO throw error within traced
                    # if $(Tf(typemin(Ti))) <= x < $(Tf(typemax(Ti)))
                    return Base.unsafe_trunc($Ti, x)
                    # else
                    #     throw(Base.InexactError(:trunc, $Ti, x))
                    # end
                end
            end
        end
    end
end

function Base.round(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.round(x))
end
function Base.floor(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.floor(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, Base.ceil(x))
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

for (Ti, Tf) in ((Int16, Float16), (Int32, Float32), (Int64, Float64))
    @eval begin
        Base.signbit(x::TracedRNumber{$(Ti)}) = x < 0
        Base.signbit(x::TracedRNumber{$(Tf)}) = signbit(Ops.bitcast_convert($(Ti), x))
    end
end
Base.signbit(::TracedRNumber{<:Unsigned}) = ConcretePJRTNumber(false)

Base.copysign(x::TracedRNumber, y::TracedRNumber) = ifelse(signbit(y), -1, 1) * abs(x)
function Base.copysign(x::TracedRNumber{T}, y::S) where {T,S<:Number}
    return copysign(x, TracedUtils.promote_to(TracedRNumber{S}, y))
end
function Base.copysign(x::S, y::TracedRNumber{T}) where {S<:Number,T}
    return copysign(TracedUtils.promote_to(TracedRNumber{S}, x), y)
end

end # module TracedRNumberOverrides
