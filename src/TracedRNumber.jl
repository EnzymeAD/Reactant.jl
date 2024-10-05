mutable struct TracedRNumber{T} <: RNumber{T}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}

    function TracedRNumber{T}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
    ) where {T}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == ()
        end
        return new{T}(paths, mlir_data)
    end
end

Base.eltype(::Type{TracedRNumber{T}}) where {T} = T

Base.getindex(a::TracedRNumber{T}) where {T} = a

Base.zero(::TracedRNumber{T}) where {T} = promote_to(TracedRNumber{T}, zero(T))
Base.one(::TracedRNumber{T}) where {T} = promote_to(TracedRNumber{T}, one(T))

Base.eps(::Type{TracedRNumber{T}}) where {T} = promote_to(TracedRNumber{T}, eps(T))

function Base.convert(::Type{<:TracedRNumber{T}}, x::Number) where {T}
    return promote_to(TracedRNumber{T}, T(x))
end

function Base.show(io::IOty, X::TracedRNumber{T}) where {T,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRNumber{", T, "}(", X.paths, ")")
end

Base.only(A::TracedRNumber{T}) where {T} = A

function Base.promote_rule(::Type{TracedRNumber{T}}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.promote_rule(::Type{T}, ::Type{TracedRNumber{S}}) where {T,S}
    return TracedRNumber{Base.promote_type(T, S)}
end

function Base.convert(::Type{TracedRNumber{T}}, x::Number) where {T}
    return promote_to(TracedRNumber{T}, x)
end

function promote_to(::Type{TracedRNumber{T}}, rhs) where {T}
    if isa(rhs, TracedRNumber)
        rhs isa TracedRNumber{T} && return rhs
        return TracedRNumber{T}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRNumber{T})
                ),
                1,
            ),
        )
    end
    if isa(rhs, TracedRArray{<:Any,0})
        return TracedRNumber{T}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRNumber{T})
                ),
                1,
            ),
        )
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(T(rhs)), mlir_type(TracedRNumber{T}))
        return TracedRNumber{T}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
        )
    end
    T0 = eltype(rhs)
    attr = MLIR.IR.DenseElementsAttribute(collect(rhs))
    return promote_to(
        TracedRNumber{T},
        TracedRNumber{T0}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
        ),
    )
end

promote_to(::TracedRNumber{T}, rhs) where {T} = promote_to(TracedRNumber{T}, rhs)

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
        return TracedRNumber{T}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.$(hloop)(lhs.mlir_data, rhs.mlir_data), 1
            ),
        )
    end
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
            return TracedRNumber{Bool}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$(hloop)(
                        lhs.mlir_data,
                        rhs.mlir_data;
                        comparison_direction=MLIR.API.stablehloComparisonDirectionAttrGet(
                            MLIR.IR.context(), $hlocomp
                        ),
                    ),
                    1,
                ),
            )
        end

        function $(jlop)(@nospecialize(lhs::TracedRNumber{T}), @nospecialize(rhs)) where {T}
            return $(jlop)(lhs, promote_to(lhs, rhs))
        end

        function $(jlop)(@nospecialize(lhs), @nospecialize(rhs::TracedRNumber{T})) where {T}
            return $(jlop)(promote_to(rhs, lhs), rhs)
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRNumber{T1}), @nospecialize(rhs::TracedRNumber{T2})
        ) where {T1,T2}
            commonTy = TracedRNumber{Base.promote_type(T1, T2)}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end
end

function Base.ifelse(
    @nospecialize(pred::TracedRNumber{Bool}),
    @nospecialize(x::TracedRNumber{T1}),
    @nospecialize(y::TracedRNumber{T2})
) where {T1,T2}
    return TracedRNumber{promote_type(T1, T2)}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.select(pred.mlir_data, x.mlir_data, y.mlir_data), 1
        ),
    )
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
    (:(Base.log), :log),
    (:(Base.sqrt), :sqrt),
)
    @eval function $(jlop)(@nospecialize(lhs::TracedRNumber{T})) where {T}
        return TracedRNumber{T}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1)
        )
    end
end

# XXX: Enzyme-MLIR doesn't have `abs` adjoint defined
Base.abs2(x::TracedRNumber{<:Real}) = x^2

Base.log1p(x::TracedRNumber{T}) where {T} = log(x + one(T))

struct TypeCast{T<:ReactantPrimitives} <: Function end

(::TypeCast{T})(x::TracedRNumber{T2}) where {T,T2} = promote_to(TracedRNumber{T}, x)

Base.float(x::TracedRNumber{T}) where {T} = promote_to(TracedRNumber{float(T)}, x)
