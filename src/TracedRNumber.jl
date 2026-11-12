module TracedRNumberOverrides

using ..Reactant: Reactant, TracedRNumber, TracedRArray, Ops, unwrapped_eltype
using ..Ops: @opcall
using ReactantCore: ReactantCore, @trace
using Adapt: Adapt

# This isn't technically necessary in this module, but this type used to be
# defined in this module so we keep this alias here for compatibility.  TODO(#2236):
# can be removed in future breaking version of Reactant.
const TracedStepRangeLen = Reactant.TracedStepRangeLen

import Base.TwicePrecision

ReactantCore.is_traced(::TracedRNumber, seen) = true
ReactantCore.is_traced(::TracedRNumber) = true

Base.to_index(x::TracedRNumber{<:Integer}) = x

Base.precision(x::TracedRNumber{T}; kwargs...) where {T} = precision(T; kwargs...)

Base.zero(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = Reactant.promote_to(TracedRNumber{T}, one(T))
Base.collect(x::TracedRNumber{T}) where {T} = TracedRArray{T,0}((), x.mlir_data, ())

Base.copy(x::TracedRNumber{T}) where {T} = TracedRNumber{T}((), x.mlir_data)

Base.eps(::Type{TracedRNumber{T}}) where {T} = eps(T)
Base.eps(x::TracedRNumber{T}) where {T} = eps(typeof(x))

function Base.typemin(::Type{<:TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemin(T))
end
Base.typemin(x::TracedRNumber{T}) where {T} = typemin(typeof(x))

function Base.typemax(::Type{<:TracedRNumber{T}}) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemax(T))
end
Base.typemax(x::TracedRNumber{T}) where {T} = typemax(typeof(x))

Base.floatmin(::Type{Reactant.TracedRNumber{T}}) where {T} = floatmin(T)
Base.floatmax(::Type{Reactant.TracedRNumber{T}}) where {T} = floatmax(T)

function Base.nextfloat(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return @opcall next_after(x, typemax(x))
end

function Base.prevfloat(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return @opcall next_after(x, typemin(x))
end

function Base.rtoldefault(T::Type{<:TracedRNumber})
    return T(Base.rtoldefault(unwrapped_eltype(T)))
end

function Base.isfinite(x::TracedRNumber{<:Complex})
    return isfinite(real(x)) & isfinite(imag(x))
end
Base.isfinite(x::TracedRNumber{<:AbstractFloat}) = @opcall is_finite(x)

function Base.isnan(x::TracedRNumber{<:Complex})
    return isnan(real(x)) | isnan(imag(x))
end
function Base.isnan(x::TracedRNumber{T}) where {T<:AbstractFloat}
    return !isfinite(x) & (x != typemax(T)) & (x != typemin(T))
end

Base.isinf(x::TracedRNumber{<:Complex}) = isinf(real(x)) | isinf(imag(x))
Base.isinf(x::TracedRNumber{<:AbstractFloat}) = @opcall is_inf(x)
Base.isinf(::TracedRNumber{<:Integer}) = false

# Base's generic `isapprox` requires both arguments to be `Real`, which
# `TracedRNumber` (<: RNumber, not Real) never satisfies -- dispatch falls
# through with no matching method. Reimplement the same algorithm here using
# `&`/`|` instead of `&&`/`||`, since traced comparisons return a traced
# boolean rather than a native `Bool` usable in short-circuiting control flow.
function Base.isapprox(
    x::TracedRNumber{T},
    y::Union{Real,TracedRNumber{<:AbstractFloat}};
    atol::Real=0,
    rtol::Real=(atol > 0 ? zero(T) : sqrt(eps(T))),
    nans::Bool=false,
    norm::Function=abs,
) where {T<:AbstractFloat}
    return (x == y) |
           (
               isfinite(x) &
               isfinite(y) &
               (norm(x - y) <= max(atol, rtol * max(norm(x), norm(y))))
           ) |
           (nans & isnan(x) & isnan(y))
end
function Base.isapprox(x::Real, y::TracedRNumber{<:AbstractFloat}; kwargs...)
    return isapprox(y, x; kwargs...)
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

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{S}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(
    T::Type{<:AbstractIrrational}, ::Type{Reactant.TracedRNumber{S}}
) where {S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(
    ::Type{Reactant.TracedRNumber{S}}, T::Type{<:AbstractIrrational}
) where {S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(::Type{Nothing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Nothing}) where {T}
    return Union{Nothing,TracedRNumber{T}}
end

function Base.promote_rule(::Type{Missing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{S}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{Missing}) where {T}
    return Union{Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    ::Type{Union{Nothing,Missing}}, ::Type{TracedRNumber{S}}
) where {S}
    return Union{Nothing,Missing,TracedRNumber{S}}
end

function Base.promote_rule(
    ::Type{TracedRNumber{T}}, ::Type{Union{Nothing,Missing}}
) where {T}
    return Union{Nothing,Missing,TracedRNumber{T}}
end

function Base.promote_rule(
    T::Type{>:Union{Nothing,Missing}}, ::Type{TracedRNumber{S}}
) where {S}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(
    ::Type{TracedRNumber{T}}, S::Type{>:Union{Nothing,Missing}}
) where {T}
    T2 = nonmissingtype(Base.nonnothingtype(promote_rule(T, S)))
    return Union{Nothing,Missing,TracedRNumber{T2}}
end

function Base.promote_rule(T::Type{>:Missing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, S::Type{>:Missing}) where {T}
    return Union{Missing,TracedRNumber{nonmissingtype(promote_type(T, S))}}
end

function Base.promote_rule(T::Type{>:Nothing}, ::Type{TracedRNumber{S}}) where {S}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(S, T))}}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, S::Type{>:Nothing}) where {T}
    return Union{Nothing,TracedRNumber{Base.nonnothingtype(promote_type(T, S))}}
end

function Base.promote_rule(::Type{TwicePrecision{T}}, ::Type{TracedRNumber{S}}) where {T,S}
    return TwicePrecision{Base.promote_type(T, TracedRNumber{S})}
end

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{TwicePrecision{S}}) where {T,S}
    return TwicePrecision{Base.promote_type(TracedRNumber{T}, S)}
end

# NOTE: This is inconsistent with the behavior of `convert` but we do it since it is a very
#       common usecase
TracedRNumber{T}(x::TracedRNumber{T}) where {T} = x
function TracedRNumber{T}(x::TracedRNumber) where {T}
    return Reactant.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end
function TracedRNumber{T}(x::Number) where {T}
    return Reactant.promote_to(TracedRNumber{unwrapped_eltype(T)}, x)
end

for T in Base.uniontypes(Reactant.ReactantFloat8)
    @eval TracedRNumber{T}(x::$T) where {T} = Reactant.promote_to(TracedRNumber{T}, x)
end

for (aT, bT) in (
    (TracedRNumber{<:Real}, Real),
    (Real, TracedRNumber{<:Real}),
    (TracedRNumber{<:Real}, TracedRNumber{<:Real}),
)
    @eval function Base.Complex(a::$aT, b::$bT)
        T = promote_type(unwrapped_eltype(a), unwrapped_eltype(b))
        a = Reactant.promote_to(TracedRNumber{T}, a)
        b = Reactant.promote_to(TracedRNumber{T}, b)
        return @opcall complex(a, b)
    end
end

Base.Complex(x::TracedRNumber{<:Real}) = @opcall complex(x, zero(x))
Base.Complex(x::TracedRNumber{<:Complex}) = x

# Base.complex
Base.complex(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{complex(T)}
Base.complex(x::TracedRNumber{<:Real}) = complex(x, zero(x))
function Base.complex(x::TracedRNumber{<:Real}, y::TracedRNumber{<:Real})
    T = promote_type(unwrapped_eltype(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRNumber{<:Real}, y::Real)
    T = promote_type(unwrapped_eltype(x), typeof(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::Real, y::TracedRNumber{<:Real})
    T = promote_type(typeof(x), unwrapped_eltype(y))
    return complex(
        Reactant.promote_to(TracedRNumber{T}, x), Reactant.promote_to(TracedRNumber{T}, y)
    )
end
function Base.complex(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T<:Real}
    return @opcall complex(x, y)
end
Base.complex(x::TracedRNumber{T}) where {T<:Complex} = x

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
    @eval function $(jlop)(lhs::TracedRNumber{T}, rhs::TracedRNumber{T}) where {T}
        return @opcall $(hloop)(lhs, rhs)
    end
end

for jlop in (:(Base.:+), :(Base.:-))
    @eval function $(jlop)(x::TracedRNumber{Bool}, y::TracedRNumber{Bool})
        return $(jlop)(
            Reactant.promote_to(TracedRNumber{Int}, x),
            Reactant.promote_to(TracedRNumber{Int}, y),
        )
    end
    @eval $(jlop)(x::TracedRNumber{Bool}) =
        $(jlop)(Reactant.promote_to(TracedRNumber{Int}, x))
end

for (wrapop, jlop) in ((Symbol("+%"), :+), (Symbol("-%"), :-))
    isdefined(Base, wrapop) || continue
    @eval Base.$(wrapop)(x::TracedRNumber{Bool}, y::TracedRNumber{Bool}) = $(jlop)(x, y)
    @eval Base.$(wrapop)(x::TracedRNumber{Bool}) = $(jlop)(x)
end

function Base.rem(x::TracedRNumber, y::TracedRNumber, ::typeof(Base.RoundFromZero))
    return ifelse(
        signbit(x) == signbit(y), rem(x, y, Base.RoundUp), rem(x, y, Base.RoundDown)
    )
end

function Base.:*(x::TracedRNumber{T}, z::Complex{Bool}) where {T<:Real}
    # this is to support multiplication by im (Complex{Bool}(false, true))
    z_re, z_im = real(z), imag(z)
    res_re = z_re ? x : zero(x)
    res_im = z_im ? x : zero(x)
    return Complex(res_re, res_im)
end
Base.:*(z::Complex{Bool}, x::TracedRNumber{T}) where {T<:Real} = x * z

# Based on https://github.com/JuliaLang/julia/blob/39255d47db7657950ff1c82137ecec5a70bae622/base/float.jl#L608-L617
function Base.mod(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    r = rem(x, y)
    return ifelse(r == 0, copysign(r, y), ifelse((r > 0) ⊻ (y > 0), r + y, r))
end

function Base.mod1(
    @nospecialize(x::Reactant.TracedRNumber{T}), @nospecialize(y::Reactant.TracedRNumber{T})
) where {T}
    m = mod(x, y)
    return ifelse(m == 0, y, m)
end

for op in (:mod, :mod1, :rem)
    @eval begin
        function Base.$op(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(op)(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
        end
        function Base.$op(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(op)(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
        end
    end
end

Base.flipsign(x::TracedRNumber, y::TracedRNumber) = ifelse(y < 0, -x, x)

function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt}, y::TracedRNumber{<:Reactant.ReactantUInt}
)
    return div(x, y, RoundDown)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundToZero),
)
    return flipsign(signed(div(unsigned(abs(x)), y)), x)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundDown),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    result = flipsign(q, x)
    return ifelse(signbit(x) & has_rem, result - one(result), result)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundUp),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    result = flipsign(q, x)
    return ifelse(!signbit(x) & has_rem, result + one(result), result)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantSInt},
    y::TracedRNumber{<:Reactant.ReactantUInt},
    ::typeof(RoundFromZero),
)
    ax = unsigned(abs(x))
    q = signed(div(ax, y))
    has_rem = !iszero(rem(ax, y))
    q_adj = q + ifelse(has_rem, one(q), zero(q))
    return flipsign(q_adj, x)
end
function Base.div(
    x::TracedRNumber{<:Reactant.ReactantUInt}, y::TracedRNumber{<:Reactant.ReactantSInt}
)
    return unsigned(flipsign(signed(div(x, unsigned(abs(y)))), y))
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}), rhs, r::Base.RoundingMode
) where {T<:Integer}
    return div(lhs, Reactant.promote_to(TracedRNumber{T}, rhs), r)
end
function Base.div(
    lhs, @nospecialize(rhs::TracedRNumber{T}), r::Base.RoundingMode
) where {T<:Integer}
    return div(Reactant.promote_to(TracedRNumber{T}, lhs), rhs, r)
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T1}),
    @nospecialize(rhs::TracedRNumber{T2}),
    r::Base.RoundingMode,
) where {T1<:Integer,T2<:Integer}
    T = promote_type(T1, T2)
    return div(
        Reactant.promote_to(TracedRNumber{T}, lhs),
        Reactant.promote_to(TracedRNumber{T}, rhs),
        r,
    )
end

function Base.div(@nospecialize(lhs::TracedRNumber{T}), rhs) where {T}
    return @opcall divide(lhs, Reactant.promote_to(TracedRNumber{T}, rhs))
end
function Base.div(lhs, @nospecialize(rhs::TracedRNumber{T})) where {T}
    return @opcall divide(Reactant.promote_to(TracedRNumber{T}, lhs), rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
) where {T1,T2}
    T = promote_type(T1, T2)
    return @opcall divide(
        Reactant.promote_to(TracedRNumber{T}, lhs),
        Reactant.promote_to(TracedRNumber{T}, rhs),
    )
end

function Base.div(
    @nospecialize(lhs::TracedRNumber{T}),
    @nospecialize(rhs::TracedRNumber{T}),
    ::typeof(RoundToZero),
) where {T<:Integer}
    return @opcall divide(lhs, rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T}),
    @nospecialize(rhs::TracedRNumber{T}),
    ::typeof(RoundDown),
) where {T<:Integer}
    return @opcall divide(lhs, rhs)
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T}),
    @nospecialize(rhs::TracedRNumber{T}),
    ::typeof(RoundUp),
) where {T<:Integer}
    q = div(lhs, rhs)  # truncation (RoundToZero)
    return q + (!iszero(rem(lhs, rhs)) & (signbit(lhs) == signbit(rhs)))
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T}),
    @nospecialize(rhs::TracedRNumber{T}),
    ::typeof(RoundFromZero),
) where {T<:Integer}
    return ifelse(
        signbit(lhs) == signbit(rhs), div(lhs, rhs, RoundUp), div(lhs, rhs, RoundDown)
    )
end
function Base.div(
    @nospecialize(lhs::TracedRNumber{T}),
    @nospecialize(rhs::TracedRNumber{T}),
    r::Union{typeof(RoundNearest),typeof(RoundNearestTiesAway),typeof(RoundNearestTiesUp)},
) where {T<:Integer}
    return divrem(lhs, rhs, r)[1]
end

function Base.div(
    @nospecialize(x::TracedRNumber{T}), @nospecialize(y::TracedRNumber{T}), r::RoundingMode
) where {T<:AbstractFloat}
    return round(div(x, y), r)
end

Base.div(@nospecialize(lhs::TracedRNumber{<:Integer}), ::Missing, ::RoundingMode) = missing
Base.div(::Missing, @nospecialize(rhs::TracedRNumber{<:Integer}), ::RoundingMode) = missing

function Base.divrem(
    @nospecialize(a::TracedRNumber{T}),
    @nospecialize(b::TracedRNumber{T}),
    r::Union{typeof(RoundUp),typeof(RoundDown),typeof(RoundToZero)},
) where {T<:Integer}
    if r === RoundToZero
        d = div(a, b)
        return (d, a - d * b)
    elseif r === RoundDown
        d = fld(a, b)
        return (d, a - d * b)
    elseif r === RoundUp
        d = div(a, b, r)
        return (d, a - d * b)
    end
end

function Base.divrem(
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T}),
    ::typeof(RoundNearest),
) where {T<:Integer}
    (q, r) = divrem(x, y)
    threshold = isodd(y) | iseven(q)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + threshold, (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + threshold, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - threshold, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - threshold, (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T}),
    ::typeof(RoundNearestTiesAway),
) where {T<:Integer}
    (q, r) = divrem(x, y)
    threshold = isodd(y)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + threshold, (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + threshold, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - threshold, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - threshold, (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T}),
    ::typeof(RoundNearestTiesUp),
) where {T<:Integer}
    (q, r) = divrem(x, y)
    half_y = y ÷ 2
    # x >= 0, y >= 0
    q1, r1 = ifelse(r >= half_y + isodd(y), (q + true, r - y), (q, r))
    # x >= 0, y < 0
    q2, r2 = ifelse(r >= -half_y + true, (q - true, r + y), (q, r))
    # x < 0, y >= 0
    q3, r3 = ifelse(r <= -half_y - true, (q - true, r + y), (q, r))
    # x < 0, y < 0
    q4, r4 = ifelse(r <= half_y - isodd(y), (q + true, r - y), (q, r))
    # Combine with ifelse based on signs
    q_pos_y, r_pos_y = ifelse(y >= 0, (q1, r1), (q2, r2))
    q_neg_y, r_neg_y = ifelse(y >= 0, (q3, r3), (q4, r4))
    return ifelse(x >= 0, (q_pos_y, r_pos_y), (q_neg_y, r_neg_y))
end

function Base.divrem(
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T}),
    ::typeof(RoundFromZero),
) where {T<:Integer}
    q_up, r_up = divrem(x, y, RoundUp)
    q_down, r_down = divrem(x, y, RoundDown)
    return ifelse(signbit(x) == signbit(y), (q_up, r_up), (q_down, r_down))
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
            return @opcall compare(lhs, rhs; comparison_direction=$(hlocomp))
        end

        # ambiguity fixes
        $(jlop)(@nospecialize(lhs::TracedRNumber), @nospecialize(::Missing)) = missing
        $(jlop)(@nospecialize(::Missing), @nospecialize(rhs::TracedRNumber)) = missing

        function $(jlop)(@nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs)) where {T}
            return $(jlop)(lhs, Reactant.promote_to(lhs, rhs))
        end
        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs::Number)
        ) where {T}
            return $(jlop)(lhs, Reactant.promote_to(lhs, rhs))
        end

        function $(jlop)(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber{T})) where {T}
            return $(jlop)(Reactant.promote_to(rhs, lhs), rhs)
        end
        function $(jlop)(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRNumber{T})
        ) where {T}
            return $(jlop)(Reactant.promote_to(rhs, lhs), rhs)
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
        ) where {T1,T2}
            commonTy = TracedRNumber{Base.promote_type(T1, T2)}
            lhs = Reactant.promote_to(commonTy, lhs)
            rhs = Reactant.promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end
end

function Base.ifelse(@nospecialize(pred::TracedRNumber{Bool}), x::Number, y::Number)
    T1 = unwrapped_eltype(x)
    T2 = unwrapped_eltype(y)
    if T1 != T2
        @warn "`ifelse` with different element-types in Reactant works by promoting the \
                element-type to the common type. This is semantically different from the \
                behavior of `ifelse` in Base. Use with caution" maxlog = 1
    end
    T = promote_type(T1, T2)
    return ifelse(
        pred,
        Reactant.promote_to(TracedRNumber{T}, x),
        Reactant.promote_to(TracedRNumber{T}, y),
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
        Reactant.promote_to(TracedRNumber{T}, x),
        Reactant.promote_to(TracedRNumber{T}, y),
    )
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T}),
    @nospecialize(y::TracedRNumber{T})
) where {T}
    return @opcall select(pred, x, y)
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::Tuple),
    @nospecialize(y::Tuple)
)
    @assert length(x) == length(y)
    ntuple(Val(length(x))) do i
        Base.ifelse(pred, x[i], y[i])
    end
end

function Base.:*(
    x::Base.TwicePrecision{T}, y::Base.TwicePrecision{T}
) where {T<:TracedRNumber}
    zh, zl = Base.mul12(x.hi, y.hi)
    hi, lo = Base.canonicalize2(zh, (x.hi * y.lo + x.lo * y.hi) + zl)
    hi = ifelse(iszero(zh) | !isfinite(zh), zh, hi)
    lo = ifelse(iszero(zl) | !isfinite(zl), zl, lo)

    return Base.TwicePrecision{T}(hi, lo)
end

function Base.:+(
    x::Base.TwicePrecision{T}, y::Base.TwicePrecision{T}
) where {T<:TracedRNumber}
    r = x.hi + y.hi
    @trace s = if abs(x.hi) > abs(y.hi)
        begin
            (((x.hi - r) + y.hi) + y.lo) + x.lo
        end
    else
        begin
            (((y.hi - r) + x.hi) + x.lo) + y.lo
        end
    end
    return Base.TwicePrecision(Base.canonicalize2(r, s)...)
end

function Base.:*(x::TwicePrecision, v::TracedRNumber)
    @trace result = if v == 0
        TwicePrecision(x.hi * v, x.lo * v)
    else
        x * TwicePrecision(oftype(x.hi * v, v))
    end
    return result
end

for (T1, T2) in Iterators.product((Bool, Integer), (Bool, Integer))
    @eval begin
        function Base.:&(x::TracedRNumber{A}, y::TracedRNumber{B}) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall and(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.:&(x::TracedRNumber{A}, y::B) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall and(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.:&(x::A, y::TracedRNumber{B}) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall and(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.:|(x::TracedRNumber{A}, y::TracedRNumber{B}) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall or(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.:|(x::TracedRNumber{A}, y::B) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall or(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.:|(x::A, y::TracedRNumber{B}) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall or(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.xor(
            x::TracedRNumber{A}, y::TracedRNumber{B}
        ) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.xor(x::TracedRNumber{A}, y::B) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
        function Base.xor(x::A, y::TracedRNumber{B}) where {A<:$(T1),B<:$(T2)}
            C = promote_type(A, B)
            return @opcall xor(
                Reactant.promote_to(TracedRNumber{C}, x),
                Reactant.promote_to(TracedRNumber{C}, y),
            )
        end
    end
end
Base.:!(x::TracedRNumber{<:Union{Bool,Integer}}) = @opcall not(x)

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRNumber{T}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
    (:(Base.abs), :abs),
    (:(Base.:-), :negate),
    (:(Base.sin), :sine),
    (:(Base.sinh), :sinh),
    (:(Base.cos), :cosine),
    (:(Base.cosh), :cosh),
    (:(Base.tan), :tan),
    (:(Base.tanh), :tanh),
    (:(Base.FastMath.tanh_fast), :tanh),
    (:(Base.exp), :exponential),
    (:(Base.FastMath.exp_fast), :exponential),
    (:(Base.expm1), :exponential_minus_one),
    (:(Base.log), :log),
    (:(Base.log1p), :log_plus_one),
    (:(Base.sqrt), :sqrt),
    (:(Base.cbrt), :cbrt),
    (:(Base.acos), :acos),
    (:(Base.acosh), :acosh),
    (:(Base.asin), :asin),
    (:(Base.asinh), :asinh),
    (:(Base.atan), :atan),
    (:(Base.atanh), :atanh),
    (:(Base.sign), :sign),
    (:(Base.conj), :conj),
    (:(Base.real), :real),
    (:(Base.imag), :imag),
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber)) = @opcall $(hloop)(lhs)
end

# Degree-based trigonometric wrappers for TracedRNumber
# These convert to radians internally so Reactant can lower to
# StableHLO-supported radian trigonometric operations.

Base.sind(x::TracedRNumber) = sin(deg2rad(x))
Base.cosd(x::TracedRNumber) = cos(deg2rad(x))
Base.tand(x::TracedRNumber) = tan(deg2rad(x))
Base.cscd(x::TracedRNumber) = 1 / sind(x)
Base.secd(x::TracedRNumber) = 1 / cosd(x)
Base.cotd(x::TracedRNumber) = 1 / tand(x)

Base.asind(x::TracedRNumber) = rad2deg(asin(x))
Base.acosd(x::TracedRNumber) = rad2deg(acos(x))
Base.atand(x::TracedRNumber) = rad2deg(atan(x))

Base.atan(y::TracedRNumber, x::TracedRNumber) = @opcall atan2(y, x)
Base.atand(y::TracedRNumber, x::TracedRNumber) = rad2deg(atan(y, x))

Base.hypot(x::TracedRNumber{T}, y::TracedRNumber{T}) where {T} = @opcall hypot(x, y)
Base.hypot(x::TracedRNumber{T}, y) where {T} = Base.hypot(x, Reactant.promote_to(x, y))
function Base.hypot(x::TracedRNumber{T}, y::Number) where {T}
    return Base.hypot(x, Reactant.promote_to(x, y))
end
Base.hypot(x, y::TracedRNumber{T}) where {T} = Base.hypot(Reactant.promote_to(y, x), y)
function Base.hypot(x::Number, y::TracedRNumber{T}) where {T}
    return Base.hypot(Reactant.promote_to(y, x), y)
end

function Base.hypot(x::TracedRNumber{T1}, y::TracedRNumber{T2}) where {T1,T2}
    commonTy = TracedRNumber{Base.promote_type(T1, T2)}
    return Base.hypot(Reactant.promote_to(commonTy, x), Reactant.promote_to(commonTy, y))
end

Base.hypot(::Missing, ::TracedRNumber) = missing
Base.hypot(::TracedRNumber, ::Missing) = missing

Base.acscd(x::TracedRNumber) = rad2deg(asin(1 / x))
Base.asecd(x::TracedRNumber) = rad2deg(acos(1 / x))
Base.acotd(x::TracedRNumber) = rad2deg(atan(1 / x))

for (jlop, hloop) in (
    (:(Base.sin), :sine),
    (:(Base.sinh), :sinh),
    (:(Base.cos), :cosine),
    (:(Base.cosh), :cosh),
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
)
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{<:Integer})) =
        @opcall $(hloop)(float(lhs))
end

for (jlop, hloop) in
    ((:(Base.sinpi), :sine), (:(Base.cospi), :cosine), (:(Base.tanpi), :tan))
    @eval $(jlop)(@nospecialize(lhs::TracedRNumber{T})) where {T} =
        @opcall $(hloop)(T(π) * lhs)
end

function Base.sincospi(x::TracedRNumber{T}) where {T}
    return @opcall(sine(T(π) * x)), @opcall(cosine(T(π) * x))
end

function Base.cispi(x::TracedRNumber)
    s, c = sincospi(x)
    return complex(c, s)
end

function Base.cis(x::TracedRNumber)
    s, c = sincos(x)
    return complex(c, s)
end

function Base.sinc(x::TracedRNumber)
    r = ifelse(iszero(x), one(x), sinpi(x) / (pi * x))
    return ifelse(isinf(x), zero(x), r)
end

function Base.sinc(x::TracedRNumber{<:Complex{T}}) where {T}
    r1 = ifelse(
        abs(x) < Base.Math._sinc_threshold(T),
        evalpoly(x^2, (T(1), -T(pi)^2 / 6, T(pi)^4 / 120)),
        sinpi(x) / (pi * x),
    )
    return ifelse(isinf(real(x)), zero(x), r1)
end

@noinline Base.Math.log10(x::TracedRNumber) = Base.Math._log(x, Val(10), :log10)
@noinline Base.Math.log2(x::TracedRNumber) = Base.Math._log(x, Val(2), :log2)
Base.Math._log(x::TracedRNumber, base, ::Symbol) = log(x) / log(Reactant._unwrap_val(base))

Base.isreal(x::TracedRNumber) = iszero(imag(x))
Base.isreal(::TracedRNumber{<:Real}) = true

Base.isinteger(x::TracedRNumber{<:Integer}) = true
Base.isinteger(x::TracedRNumber{<:AbstractFloat}) = x - trunc(x) == zero(x)

Base.isodd(x::TracedRNumber) = isodd(real(x))
function Base.isodd(x::TracedRNumber{<:Real})
    return (
        isinteger(x) &
        !iszero(
            rem(
                Reactant.promote_to(TracedRNumber{Int}, x),
                Reactant.promote_to(TracedRNumber{Int}, 2),
            ),
        )
    )
end

Base.iseven(x::TracedRNumber) = iseven(real(x))
function Base.iseven(x::TracedRNumber{<:Real})
    return (
        isinteger(x) & iszero(
            rem(
                Reactant.promote_to(TracedRNumber{Int}, x),
                Reactant.promote_to(TracedRNumber{Int}, 2),
            ),
        )
    )
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval function Base.clamp(x::TracedRNumber, min::$(minT), max::$(maxT))
        T = promote_type(unwrapped_eltype(x), unwrapped_eltype(min), unwrapped_eltype(max))
        return @opcall clamp(
            Reactant.promote_to(TracedRNumber{T}, min),
            Reactant.promote_to(TracedRNumber{T}, x),
            Reactant.promote_to(TracedRNumber{T}, max),
        )
    end
end

function Base.float(x::TracedRNumber{T}) where {T}
    return Reactant.promote_to(TracedRNumber{float(T)}, x)
end

using Reactant: ReactantFloat, ReactantInt

Base.round(A::TracedRNumber{<:ReactantFloat}) = @opcall round_nearest_even(A)
Base.round(A::TracedRNumber{<:ReactantInt}) = A
function Base.round(A::TracedRNumber{<:ReactantFloat}, ::typeof(RoundNearest))
    return @opcall round_nearest_even(A)
end
function Base.round(A::TracedRNumber{<:ReactantFloat}, ::typeof(RoundNearestTiesAway))
    return @opcall round_nearest_afz(A)
end
Base.round(A::TracedRNumber{<:ReactantFloat}, ::typeof(RoundUp)) = ceil(A)
Base.round(A::TracedRNumber{<:ReactantFloat}, ::typeof(RoundDown)) = floor(A)
function Base.round(A::TracedRNumber{<:ReactantFloat}, ::typeof(RoundToZero))
    A_truncated = @opcall convert(TracedRNumber{Int}, A)
    return copysign(
        @opcall(convert(TracedRNumber{Reactant.unwrapped_eltype(A)}, A_truncated)), A
    )
end

Base.floor(A::TracedRNumber{<:ReactantFloat}) = @opcall floor(A)
Base.floor(A::TracedRNumber{<:ReactantInt}) = A
Base.ceil(A::TracedRNumber{<:ReactantFloat}) = @opcall ceil(A)
Base.ceil(A::TracedRNumber{<:ReactantInt}) = A

function Base.unsafe_trunc(
    T::Type{<:Reactant.ReactantInt}, x::TracedRNumber{<:Reactant.ReactantFloat}
)
    return @opcall convert(TracedRNumber{T}, x)
end

for Ti in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for Tf in (Float16, Float32, Float64)
        if Ti <: Unsigned || sizeof(Ti) < sizeof(Tf)
            # Here `Tf(typemin(Ti))-1` is exact, so we can compare the lower-bound
            # directly. `Tf(typemax(Ti))+1` is either always exactly representable, or
            # rounded to `Inf` (e.g. when `Ti==UInt128 && Tf==Float32`).
            @eval begin
                function Base.trunc(::Type{$Ti}, x::TracedRNumber{$Tf})
                    # TODO(#2236) throw error within traced
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
                    # TODO(#2236) throw error within traced
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

# matches convert methods
# also determines floor, ceil, round
Base.trunc(::Type{Signed}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(Int, x)
Base.trunc(::Type{Unsigned}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(UInt, x)
Base.trunc(::Type{Integer}, x::TracedRNumber{<:Base.IEEEFloat}) = Base.trunc(Int, x)

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.TracedRNumber}
    return (T(x.hi) + T(x.lo))::T
end

function (::Type{T})(x::TwicePrecision) where {T<:Reactant.ConcreteRNumber}
    return Reactant.ConcreteRNumber(T(x.hi) + T(x.lo))::T
end

function Base.round(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, round(x))
end
function Base.floor(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, floor(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:Integer}
    return trunc(T, ceil(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where {T<:AbstractFloat}
    return @opcall convert(TracedRNumber{T}, ceil(x))
end
function Base.ceil(::Type{T}, x::TracedRNumber{<:Integer}) where {T<:Real}
    return @opcall convert(TracedRNumber{T}, x)
end

# Concatenation. Numbers in Julia are handled in a much less generic fashion than arrays
Base.vcat(x::TracedRNumber...) = Base.typed_vcat(Base.promote_eltypeof(x...), x...)
function Base.typed_vcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_vcat(T, map(Base.Fix2(Reactant.broadcast_to_size, (1,)), x)...)
end

Base.hcat(x::TracedRNumber...) = Base.typed_hcat(Base.promote_eltypeof(x...), x...)
function Base.typed_hcat(::Type{T}, x::TracedRNumber...) where {T}
    return Base.typed_hcat(T, map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), x)...)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, xs::TracedRNumber...)
    return Base.typed_hvcat(Base.promote_eltypeof(xs...), rows, xs...)
end
function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Vararg{Int}}, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvcat(T, rows, xs...)
end

function Base.hvncat(dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...)
    return Base.typed_hvncat(Base.promote_eltypeof(xs...), dims, row_first, xs...)
end
function Base.typed_hvncat(
    ::Type{T}, dims::Tuple{Vararg{Int}}, row_first::Bool, xs::TracedRNumber...
) where {T}
    xs = map(Base.Fix2(Reactant.broadcast_to_size, (1, 1)), xs)
    return Base.typed_hvncat(T, dims, row_first, xs...)
end

for (Ti, Tf) in ((Int16, Float16), (Int32, Float32), (Int64, Float64))
    @eval begin
        Base.signbit(x::TracedRNumber{$(Ti)}) = x < 0
        Base.signbit(x::TracedRNumber{$(Tf)}) = signbit(@opcall(bitcast_convert($(Ti), x)))
    end
end
Base.signbit(::TracedRNumber{<:Unsigned}) = Reactant.promote_to(TracedRNumber{Bool}, false)

function Base.copysign(x::TracedRNumber, y::TracedRNumber)
    return ifelse(signbit(y), -one(x), one(x)) * abs(x)
end
function Base.copysign(x::TracedRNumber{T}, y::S) where {T,S<:Number}
    return copysign(x, Reactant.promote_to(TracedRNumber{S}, y))
end
function Base.copysign(x::S, y::TracedRNumber{T}) where {S<:Number,T}
    return copysign(Reactant.promote_to(TracedRNumber{S}, x), y)
end

function Base.zeros(::Type{<:TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(zero(T), dims)
end
function Base.zeros(::Type{<:TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(zero(T), ())
end

function Base.ones(::Type{<:TracedRNumber{T}}, dims::Dims{N}) where {T,N}
    return @opcall fill(one(T), dims)
end
function Base.ones(::Type{<:TracedRNumber{T}}, ::Tuple{}) where {T}
    return @opcall fill(one(T), ())
end

function Base.fill(v::TracedRNumber{T}, dims::Dims{N}) where {T,N}
    return @opcall fill(v, dims)
end
function Base.fill(v::TracedRNumber{T}, ::Tuple{}) where {T}
    return @opcall fill(v, ())
end

# TODO(#2236): actually perform bounds checking
function Base.checkindex(::Type{Bool}, _inds, ::TracedRNumber)
    @warn "Currently we don't perform bounds checking for TracedRNumber. This will be \
           fixed in a future version of Reactant." maxlog = 1
    return true
end

function Base.checkindex(::Type{Bool}, ::AbstractUnitRange, ::TracedRNumber)
    @warn "Currently we don't perform bounds checking for TracedRNumber. This will be \
           fixed in a future version of Reactant." maxlog = 1
    return true
end

const _REM2PI_INV_2PI = (
    0x28be_60db_9391_054a,
    0x7f09_d5f4_7d4d_3770,
    0x36d8_a566_4f10_e410,
    0x7f94_58ea_f7ae_f158,
    0x6dc9_1b8e_9093_74b8,
    0x0192_4bba_8274_6487,
    0x3f87_7ac7_2c4a_69cf,
    0xba20_8d7d_4bae_d121,
    0x3a67_1c09_ad17_df90,
    0x4e64_758e_60d4_ce7d,
    0x2721_17e2_ef7e_4a0e,
    0xc7fe_25ff_f781_6603,
    0xfbcb_c462_d682_9b47,
    0xdb4d_9fb3_c9f2_c26d,
    0xd3d1_8fd9_a797_fa8b,
    0x5d49_eeb1_faf9_7c5e,
    0xcf41_ce7d_e294_a4ba,
    0x9afe_d7ec_47e3_5742,
    0x1580_cc11_bf1e_daea,
)

const _REM2PI_PI1O2_H = 1.5707963267948966
const _REM2PI_PI1O2_L = 6.123233995736766e-17
const _REM2PI_PI2O2_H = 3.141592653589793
const _REM2PI_PI2O2_L = 1.2246467991473532e-16
const _REM2PI_PI3O2_H = 4.71238898038469
const _REM2PI_PI3O2_L = 1.8369701987210297e-16
const _REM2PI_PI4O2_H = 6.283185307179586
const _REM2PI_PI4O2_L = 2.4492935982947064e-16

@inline _rem2pi_u64(v::Integer) = Reactant.promote_to(TracedRNumber{UInt64}, UInt64(v))
@inline _rem2pi_i64(v::Integer) = Reactant.promote_to(TracedRNumber{Int64}, Int64(v))
@inline _rem2pi_f64(v::Real) = Reactant.promote_to(TracedRNumber{Float64}, Float64(v))

@inline function _rem2pi_shl(x::TracedRNumber{UInt64}, k::TracedRNumber{UInt64})
    return @opcall shift_left(x, k)
end
@inline function _rem2pi_shl(x::TracedRNumber{UInt64}, k::UInt64)
    return _rem2pi_shl(x, _rem2pi_u64(k))
end
@inline function _rem2pi_shr(x::TracedRNumber{UInt64}, k::TracedRNumber{UInt64})
    return @opcall shift_right_logical(x, k)
end
@inline function _rem2pi_shr(x::TracedRNumber{UInt64}, k::UInt64)
    return _rem2pi_shr(x, _rem2pi_u64(k))
end

@inline function _rem2pi_widemul_u64(x::TracedRNumber{UInt64}, y::TracedRNumber{UInt64})
    mask = _rem2pi_u64(0x0000_0000_ffff_ffff)
    x_lo = x & mask
    x_hi = _rem2pi_shr(x, UInt64(32))
    y_lo = y & mask
    y_hi = _rem2pi_shr(y, UInt64(32))
    p0 = x_lo * y_lo
    p1 = x_lo * y_hi + _rem2pi_shr(p0, UInt64(32))
    p2 = x_hi * y_lo + (p1 & mask)
    lo = _rem2pi_shl(p2, UInt64(32)) | (p0 & mask)
    hi = x_hi * y_hi + _rem2pi_shr(p1, UInt64(32)) + _rem2pi_shr(p2, UInt64(32))
    return hi, lo
end

@inline function _rem2pi_shr128_to_u64(
    x_hi::TracedRNumber{UInt64}, x_lo::TracedRNumber{UInt64}, k::TracedRNumber{Int64}
)
    z_i64 = _rem2pi_i64(0)
    c64 = _rem2pi_i64(64)
    k_ge_64 = k >= c64
    s_hi = @opcall bitcast_convert(UInt64, ifelse(k_ge_64, k - c64, z_i64))
    from_ge64 = _rem2pi_shr(x_hi, s_hi)
    k_zero = k == z_i64
    s1 = @opcall bitcast_convert(UInt64, ifelse(k_zero | k_ge_64, z_i64, k))
    s2 = @opcall bitcast_convert(UInt64, ifelse(k_zero | k_ge_64, z_i64, c64 - k))
    from_lt64 = ifelse(k_zero, x_lo, _rem2pi_shr(x_lo, s1) | _rem2pi_shl(x_hi, s2))
    return ifelse(k_ge_64, from_ge64, from_lt64)
end

@inline function _rem2pi_shl128(
    x_hi::TracedRNumber{UInt64}, x_lo::TracedRNumber{UInt64}, k::TracedRNumber{Int64}
)
    z_i64 = _rem2pi_i64(0)
    c64 = _rem2pi_i64(64)
    k_ge_64 = k >= c64
    s_lo = @opcall bitcast_convert(UInt64, ifelse(k_ge_64, k - c64, z_i64))
    hi_ge = _rem2pi_shl(x_lo, s_lo)
    k_zero = k == z_i64
    s1 = @opcall bitcast_convert(UInt64, ifelse(k_zero | k_ge_64, z_i64, k))
    s2 = @opcall bitcast_convert(UInt64, ifelse(k_zero | k_ge_64, z_i64, c64 - k))
    hi_lt = ifelse(k_zero, x_hi, _rem2pi_shl(x_hi, s1) | _rem2pi_shr(x_lo, s2))
    lo_lt = ifelse(k_zero, x_lo, _rem2pi_shl(x_lo, s1))
    return ifelse(k_ge_64, hi_ge, hi_lt), ifelse(k_ge_64, _rem2pi_u64(UInt64(0)), lo_lt)
end

@inline function _rem2pi_fromfraction_u64(
    f_hi::TracedRNumber{UInt64}, f_lo::TracedRNumber{UInt64}
)
    u0 = _rem2pi_u64(UInt64(0))
    u1 = _rem2pi_u64(UInt64(1))
    is_zero = (f_hi == u0) & (f_lo == u0)
    f_neg = _rem2pi_shr(f_hi, UInt64(63)) != u0
    s = ifelse(f_neg, _rem2pi_u64(0x8000_0000_0000_0000), u0)
    nx_lo = (@opcall not(f_lo)) + u1
    nx_hi = (@opcall not(f_hi)) + ifelse(nx_lo == u0, u1, u0)
    x_hi = ifelse(f_neg, nx_hi, f_hi)
    x_lo = ifelse(f_neg, nx_lo, f_lo)

    clz_hi = @opcall bitcast_convert(Int64, @opcall(count_leading_zeros(x_hi)))
    clz_lo = @opcall bitcast_convert(Int64, @opcall(count_leading_zeros(x_lo)))
    lz = ifelse(x_hi != u0, clz_hi, _rem2pi_i64(64) + clz_lo)
    n1 = ifelse(is_zero, _rem2pi_i64(64), _rem2pi_i64(128) - lz)

    z_i64 = _rem2pi_i64(0)
    shift1 = n1 - _rem2pi_i64(26)
    pos_sh1 = ifelse(shift1 >= z_i64, shift1, z_i64)
    neg_sh1 = ifelse(shift1 < z_i64, -shift1, z_i64)
    _, shl1_lo = _rem2pi_shl128(x_hi, x_lo, neg_sh1)
    m1_raw = ifelse(shift1 >= z_i64, _rem2pi_shr128_to_u64(x_hi, x_lo, pos_sh1), shl1_lo)
    m1 = _rem2pi_shl(m1_raw, UInt64(27))
    d1 = _rem2pi_shl(@opcall(bitcast_convert(UInt64, n1 + _rem2pi_i64(893))), UInt64(52))
    z1 = ifelse(is_zero, _rem2pi_f64(0.0), @opcall(bitcast_convert(Float64, s | (d1 + m1))))

    shift_sub = n1 - _rem2pi_i64(53)
    pos_sub = ifelse(shift_sub >= z_i64, shift_sub, z_i64)
    neg_sub = ifelse(shift_sub < z_i64, -shift_sub, z_i64)
    sh_sub_hi, sh_sub_lo = _rem2pi_shl128(u0, m1, pos_sub)
    neg_sub_u = @opcall bitcast_convert(UInt64, neg_sub)
    sub_hi = ifelse(shift_sub >= z_i64, sh_sub_hi, u0)
    sub_lo = ifelse(shift_sub >= z_i64, sh_sub_lo, _rem2pi_shr(m1, neg_sub_u))
    borrow = ifelse(x_lo < sub_lo, u1, u0)
    x2_lo = x_lo - sub_lo
    x2_hi = x_hi - sub_hi - borrow
    is_z2_zero = is_zero | ((x2_hi == u0) & (x2_lo == u0))

    clz2_hi = @opcall bitcast_convert(Int64, @opcall(count_leading_zeros(x2_hi)))
    clz2_lo = @opcall bitcast_convert(Int64, @opcall(count_leading_zeros(x2_lo)))
    lz2 = ifelse(x2_hi != u0, clz2_hi, _rem2pi_i64(64) + clz2_lo)
    n2 = ifelse(is_z2_zero, _rem2pi_i64(64), _rem2pi_i64(128) - lz2)
    shift2 = n2 - _rem2pi_i64(53)
    pos_sh2 = ifelse(shift2 >= z_i64, shift2, z_i64)
    neg_sh2 = ifelse(shift2 < z_i64, -shift2, z_i64)
    _, shl2_lo = _rem2pi_shl128(x2_hi, x2_lo, neg_sh2)
    m2 = ifelse(shift2 >= z_i64, _rem2pi_shr128_to_u64(x2_hi, x2_lo, pos_sh2), shl2_lo)
    d2 = _rem2pi_shl(@opcall(bitcast_convert(UInt64, n2 + _rem2pi_i64(893))), UInt64(52))
    z2 = ifelse(
        is_z2_zero, _rem2pi_f64(0.0), @opcall(bitcast_convert(Float64, s | (d2 + m2)))
    )
    return z1, z2
end

@inline function _rem2pi_lookup_inv2pi(i::TracedRNumber{Int64})
    acc = _rem2pi_u64(UInt64(0))
    for k in 1:19
        acc = ifelse(i == _rem2pi_i64(Int64(k)), _rem2pi_u64(_REM2PI_INV_2PI[k]), acc)
    end
    return acc
end

@inline function _rem2pi_paynehanek(x::TracedRNumber{Float64})
    x_ng = @opcall ignore_derivatives(x)
    one_f64 = _rem2pi_f64(1.0)
    ax_safe = ifelse((abs(x_ng) >= one_f64) & isfinite(x_ng), x_ng, copysign(one_f64, x_ng))
    u = @opcall bitcast_convert(UInt64, ax_safe)
    X = (u & _rem2pi_u64(0x000f_ffff_ffff_ffff)) | _rem2pi_u64(UInt64(1) << 52)
    raw_exp_u = _rem2pi_shr(u & _rem2pi_u64(0x7ff0_0000_0000_0000), UInt64(52))
    raw_exponent = @opcall bitcast_convert(Int64, raw_exp_u)
    k = raw_exponent - _rem2pi_i64(1075)

    idx_s = @opcall shift_right_arithmetic(k, _rem2pi_i64(6))
    shift_i = k - (@opcall shift_left(idx_s, _rem2pi_i64(6)))
    v1 = _rem2pi_lookup_inv2pi(idx_s + _rem2pi_i64(1))
    v2 = _rem2pi_lookup_inv2pi(idx_s + _rem2pi_i64(2))
    v3 = _rem2pi_lookup_inv2pi(idx_s + _rem2pi_i64(3))
    v4 = _rem2pi_lookup_inv2pi(idx_s + _rem2pi_i64(4))

    u0 = _rem2pi_u64(UInt64(0))
    u1 = _rem2pi_u64(UInt64(1))
    z_i64 = _rem2pi_i64(0)
    sh_zero = shift_i == z_i64
    s_l = @opcall bitcast_convert(UInt64, ifelse(sh_zero, z_i64, shift_i))
    s_r = @opcall bitcast_convert(UInt64, ifelse(sh_zero, z_i64, _rem2pi_i64(64) - shift_i))
    a1_sh = ifelse(idx_s < z_i64, u0, _rem2pi_shl(v1, s_l)) | _rem2pi_shr(v2, s_r)
    a2_sh = _rem2pi_shl(v2, s_l) | _rem2pi_shr(v3, s_r)
    a3_sh = _rem2pi_shl(v3, s_l) | _rem2pi_shr(v4, s_r)

    a1 = ifelse(sh_zero, v1, a1_sh)
    a2 = ifelse(sh_zero, v2, a2_sh)
    a3 = ifelse(sh_zero, v3, a3_sh)

    w1_hi = X * a1
    w2_hi, w2_lo = _rem2pi_widemul_u64(X, a2)
    w3_hi, _ = _rem2pi_widemul_u64(X, a3)

    w_lo = w2_lo + w3_hi
    carry = ifelse(w_lo < w2_lo, u1, u0)
    w_hi = w1_hi + w2_hi + carry

    neg_lo = (@opcall not(w_lo)) + u1
    neg_hi = (@opcall not(w_hi)) + ifelse(neg_lo == u0, u1, u0)
    x_neg = ax_safe < _rem2pi_f64(0.0)
    w_hi = ifelse(x_neg, neg_hi, w_hi)
    w_lo = ifelse(x_neg, neg_lo, w_lo)

    q_top = @opcall bitcast_convert(Int64, _rem2pi_shr(w_hi, UInt64(61)))
    q = @opcall shift_right_arithmetic(q_top + _rem2pi_i64(1), _rem2pi_i64(1))
    f_hi = _rem2pi_shl(w_hi, UInt64(2)) | _rem2pi_shr(w_lo, UInt64(62))
    f_lo = _rem2pi_shl(w_lo, UInt64(2))

    z_hi, z_lo = _rem2pi_fromfraction_u64(f_hi, f_lo)

    pio2 = _rem2pi_f64(1.5707963267948966)
    pio2_hi = _rem2pi_f64(1.5707963407039642)
    pio2_lo = _rem2pi_f64(-1.3909067614167116e-8)
    y_hi_ng = (z_hi + z_lo) * pio2
    y_lo = (((z_hi * pio2_hi - y_hi_ng) + z_hi * pio2_lo) + z_lo * pio2_hi) + z_lo * pio2_lo
    # Preserve exact unit derivative d(rem2pi(x))/dx = 1 on finite inputs without
    # perturbing the primal Float64 bits (y_hi_ng is strictly nonzero for |x| >= 2^20*pi/2).
    x_fin = ifelse(isfinite(x), x, _rem2pi_f64(0.0))
    y_hi = y_hi_ng + (x_fin - (@opcall ignore_derivatives(x_fin)))
    return q, y_hi, y_lo
end

@inline function _rem2pi_cody_waite(x::TracedRNumber{Float64})
    thresh = _rem2pi_f64(1.647099891681509e6)
    x_safe = ifelse((abs(x) < thresh) & isfinite(x), x, _rem2pi_f64(0.0))
    u_x = @opcall bitcast_convert(UInt64, @opcall(ignore_derivatives(x_safe)))
    xhp = _rem2pi_shr(u_x, UInt64(32)) & _rem2pi_u64(0x7fff_ffff)

    pio2_1 = _rem2pi_f64(1.57079632673412561417e+00)
    pio2_1t = _rem2pi_f64(6.07710050650619224932e-11)
    pio2_2 = _rem2pi_f64(6.07710050630396597660e-11)
    pio2_2t = _rem2pi_f64(2.02226624879595063154e-21)
    pio2_3 = _rem2pi_f64(2.02226624871116645580e-21)
    pio2_3t = _rem2pi_f64(8.47842766036889956997e-32)

    fn = @opcall ignore_derivatives(round(x_safe * _rem2pi_f64(0.6366197723675814)))
    r1 = x_safe - fn * pio2_1
    w1 = fn * pio2_1t
    j = @opcall bitcast_convert(Int64, _rem2pi_shr(xhp, UInt64(20)))
    y1_1 = r1 - w1
    y2_1 = (r1 - y1_1) - w1

    u_y1_1 = @opcall bitcast_convert(UInt64, @opcall(ignore_derivatives(y1_1)))
    high1 = @opcall bitcast_convert(
        Int64, _rem2pi_shr(u_y1_1, UInt64(52)) & _rem2pi_u64(0x7ff)
    )
    need2 = (j - high1) > _rem2pi_i64(16)

    t2 = r1
    w2a = fn * pio2_2
    r2 = t2 - w2a
    w2 = fn * pio2_2t - ((t2 - r2) - w2a)
    y1_2 = r2 - w2
    y2_2 = (r2 - y1_2) - w2

    u_y1_2 = @opcall bitcast_convert(UInt64, @opcall(ignore_derivatives(y1_2)))
    high2 = @opcall bitcast_convert(
        Int64, _rem2pi_shr(u_y1_2, UInt64(52)) & _rem2pi_u64(0x7ff)
    )
    need3 = need2 & ((j - high2) > _rem2pi_i64(49))

    t3 = r2
    w3a = fn * pio2_3
    r3 = t3 - w3a
    w3 = fn * pio2_3t - ((t3 - r3) - w3a)
    y1_3 = r3 - w3
    y2_3 = (r3 - y1_3) - w3

    y1 = ifelse(need3, y1_3, ifelse(need2, y1_2, y1_1))
    y2 = ifelse(need3, y2_3, ifelse(need2, y2_2, y2_1))
    n = Base.unsafe_trunc(Int64, fn)
    return n, y1, y2
end

@inline function _rem2pi_pio2_kernel(x::TracedRNumber{Float64})
    u_x = @opcall bitcast_convert(UInt64, @opcall(ignore_derivatives(x)))
    xhp = _rem2pi_shr(u_x, UInt64(32)) & _rem2pi_u64(0x7fff_ffff)
    use_cw = xhp < _rem2pi_u64(0x4139_21fb)
    n_cw, yh_cw, yl_cw = _rem2pi_cody_waite(x)
    n_ph, yh_ph, yl_ph = _rem2pi_paynehanek(x)
    return (
        ifelse(use_cw, n_cw, n_ph),
        ifelse(use_cw, yh_cw, yh_ph),
        ifelse(use_cw, yl_cw, yl_ph),
    )
end

@inline function _rem2pi_add22condh(
    xh::TracedRNumber{Float64}, xl::TracedRNumber{Float64}, yh_val::Float64, yl_val::Float64
)
    yh = _rem2pi_f64(yh_val)
    yl = _rem2pi_f64(yl_val)
    r = xh + yh
    s = ifelse(abs(xh) > abs(yh), xh - r + yh + yl + xl, yh - r + xh + xl + yl)
    return r + s
end

function Base.rem2pi(x::TracedRNumber{Float64}, ::Base.RoundingMode{:Nearest})
    n, yh, yl = _rem2pi_pio2_kernel(x)
    rem4 = n & _rem2pi_i64(3)
    res0 = yh + yl
    res1 = _rem2pi_add22condh(yh, yl, _REM2PI_PI1O2_H, _REM2PI_PI1O2_L)
    res2 = ifelse(
        yh <= _rem2pi_f64(0.0),
        _rem2pi_add22condh(yh, yl, _REM2PI_PI2O2_H, _REM2PI_PI2O2_L),
        _rem2pi_add22condh(yh, yl, -_REM2PI_PI2O2_H, -_REM2PI_PI2O2_L),
    )
    res3 = _rem2pi_add22condh(yh, yl, -_REM2PI_PI1O2_H, -_REM2PI_PI1O2_L)
    res = ifelse(
        rem4 == _rem2pi_i64(0),
        res0,
        ifelse(rem4 == _rem2pi_i64(1), res1, ifelse(rem4 == _rem2pi_i64(2), res2, res3)),
    )
    nan_f64 = _rem2pi_f64(NaN)
    return ifelse(
        isnan(x),
        x,
        ifelse(isinf(x), nan_f64, ifelse(abs(x) < _rem2pi_f64(Float64(π)), x, res)),
    )
end

function Base.rem2pi(x::TracedRNumber{Float64}, ::Base.RoundingMode{:ToZero})
    ax = abs(x)
    n, yh, yl = _rem2pi_pio2_kernel(ax)
    rem4 = n & _rem2pi_i64(3)
    res0 = ifelse(
        yh > _rem2pi_f64(0.0),
        yh + yl,
        _rem2pi_add22condh(yh, yl, _REM2PI_PI4O2_H, _REM2PI_PI4O2_L),
    )
    res1 = _rem2pi_add22condh(yh, yl, _REM2PI_PI1O2_H, _REM2PI_PI1O2_L)
    res2 = _rem2pi_add22condh(yh, yl, _REM2PI_PI2O2_H, _REM2PI_PI2O2_L)
    res3 = _rem2pi_add22condh(yh, yl, _REM2PI_PI3O2_H, _REM2PI_PI3O2_L)
    z = ifelse(
        rem4 == _rem2pi_i64(0),
        res0,
        ifelse(rem4 == _rem2pi_i64(1), res1, ifelse(rem4 == _rem2pi_i64(2), res2, res3)),
    )
    res = copysign(z, x)
    thresh = _rem2pi_f64(2.0 * Float64(π, RoundDown))
    nan_f64 = _rem2pi_f64(NaN)
    return ifelse(isnan(x), x, ifelse(isinf(x), nan_f64, ifelse(ax <= thresh, x, res)))
end

function Base.rem2pi(x::TracedRNumber{Float64}, ::Base.RoundingMode{:Down})
    n, yh, yl = _rem2pi_pio2_kernel(x)
    rem4 = n & _rem2pi_i64(3)
    zero_f64 = _rem2pi_f64(0.0)
    res0 = ifelse(
        yh > zero_f64, yh + yl, _rem2pi_add22condh(yh, yl, _REM2PI_PI4O2_H, _REM2PI_PI4O2_L)
    )
    res1 = _rem2pi_add22condh(yh, yl, _REM2PI_PI1O2_H, _REM2PI_PI1O2_L)
    res2 = _rem2pi_add22condh(yh, yl, _REM2PI_PI2O2_H, _REM2PI_PI2O2_L)
    res3 = _rem2pi_add22condh(yh, yl, _REM2PI_PI3O2_H, _REM2PI_PI3O2_L)
    res = ifelse(
        rem4 == _rem2pi_i64(0),
        res0,
        ifelse(rem4 == _rem2pi_i64(1), res1, ifelse(rem4 == _rem2pi_i64(2), res2, res3)),
    )
    small_neg = _rem2pi_add22condh(x, zero_f64, _REM2PI_PI4O2_H, _REM2PI_PI4O2_L)
    pi4 = _rem2pi_f64(_REM2PI_PI4O2_H)
    res_fin = ifelse(
        (x >= zero_f64) & (x < pi4), x, ifelse((x < zero_f64) & (x > -pi4), small_neg, res)
    )
    return ifelse(isnan(x), x, ifelse(isinf(x), _rem2pi_f64(NaN), res_fin))
end

function Base.rem2pi(x::TracedRNumber{Float64}, ::Base.RoundingMode{:Up})
    n, yh, yl = _rem2pi_pio2_kernel(x)
    rem4 = n & _rem2pi_i64(3)
    zero_f64 = _rem2pi_f64(0.0)
    res0 = ifelse(
        yh < zero_f64,
        yh + yl,
        _rem2pi_add22condh(yh, yl, -_REM2PI_PI4O2_H, -_REM2PI_PI4O2_L),
    )
    res1 = _rem2pi_add22condh(yh, yl, -_REM2PI_PI3O2_H, -_REM2PI_PI3O2_L)
    res2 = _rem2pi_add22condh(yh, yl, -_REM2PI_PI2O2_H, -_REM2PI_PI2O2_L)
    res3 = _rem2pi_add22condh(yh, yl, -_REM2PI_PI1O2_H, -_REM2PI_PI1O2_L)
    res = ifelse(
        rem4 == _rem2pi_i64(0),
        res0,
        ifelse(rem4 == _rem2pi_i64(1), res1, ifelse(rem4 == _rem2pi_i64(2), res2, res3)),
    )
    small_pos = _rem2pi_add22condh(x, zero_f64, -_REM2PI_PI4O2_H, -_REM2PI_PI4O2_L)
    pi4 = _rem2pi_f64(_REM2PI_PI4O2_H)
    res_fin = ifelse(
        (x <= zero_f64) & (x > -pi4), x, ifelse((x > zero_f64) & (x < pi4), small_pos, res)
    )
    return ifelse(isnan(x), x, ifelse(isinf(x), _rem2pi_f64(NaN), res_fin))
end

Base.rem2pi(x::TracedRNumber, r::Base.RoundingMode) = rem(x, typeof(x)(2π), r)

function Base.rem2pi(
    x::T, r::Base.RoundingMode
) where {T<:TracedRNumber{<:Union{Float16,Float32}}}
    return T(rem2pi(TracedRNumber{Float64}(x), r))
end
Base.rem2pi(x::TracedRNumber{<:Integer}, r::Base.RoundingMode) = rem2pi(float(x), r)

@static if isdefined(Base, :unchecked_oneto)
    function Base.unchecked_oneto(x::TracedRNumber{<:Integer})
        return Reactant.TracedUnitRange(one(x), x)
    end
end

Base.unsigned(x::TracedRNumber{T}) where {T<:Reactant.ReactantUInt} = x
function Base.unsigned(x::TracedRNumber{T}) where {T<:Reactant.ReactantSInt}
    return convert(TracedRNumber{unsigned(T)}, x)
end

function Base.signed(x::TracedRNumber{T}) where {T<:Reactant.ReactantUInt}
    return convert(TracedRNumber{signed(T)}, x)
end
Base.signed(x::TracedRNumber{T}) where {T<:Reactant.ReactantSInt} = x

Base.uabs(x::TracedRNumber{<:Integer}) = abs(x)
Base.uabs(x::TracedRNumber{<:Reactant.ReactantSInt}) = unsigned(abs(x))

function Base.divgcd(
    x::TracedRNumber{TX}, y::TracedRNumber{TY}
) where {TX<:Integer,TY<:Integer}
    g = gcd(Base.uabs(x), Base.uabs(y))
    return div(x, g), div(y, g)
end

# See https://github.com/EnzymeAD/Reactant.jl/issues/2476 for why this is
# implemented this way.
function Base.gcd(a::TracedRNumber{T}, b::TracedRNumber{T}) where {T<:Integer}
    a_tmp = copy(a)
    b_tmp = copy(b)
    @trace while b_tmp != 0
        t = b_tmp
        b_tmp = rem(a_tmp, b_tmp)
        a_tmp = t
    end
    return abs(a_tmp)
end

end # module TracedRNumberOverrides
