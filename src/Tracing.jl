mutable struct TracedRArray{T,N} <: RArray{T,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

Base.size(x::TracedRArray) = x.shape

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray((), A.mlir_data, size(A))

function Base.similar(x::TracedRArray{T,N}, ::Type{T2}) where {T,N,T2}
    return TracedRArray{T2,N}((), nothing, size(x))
end

function Base.promote_rule(
    ::Type{TracedRArray{T,N}}, ::Type{TracedRArray{S,N}}
) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function Base.promote_rule(A::Type{T}, B::Type{TracedRArray{S,N}}) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function promote_to(::Type{TracedRArray{T,N}}, rhs) where {T,N}
    if isa(rhs, TracedRArray)
        return TracedRArray{T,N}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRArray{T,N})
                ),
                1,
            ),
        )
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(T(rhs)), mlir_type(TracedRArray{T,N}))
        ta = TracedRArray{T,N}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
        )
        return ta
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(TracedRArray{T,N}), rhs)
    return TracedRArray{T,N}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(rhs)
    )
end

function promote_to(lhs::TracedRArray{T,N}, rhs) where {T,N}
    return promote_to(TracedRArray{T,N}, rhs)
end

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
)
    @eval begin
        function $jlop(lhs::TracedRArray{T,N}, rhs::TracedRArray{T2,N}) where {T,T2,N}
            commonTy = TracedRArray{Base.promote_type(T, T2),N}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs::TracedRArray{T,N}, rhs::TracedRArray{T,N}) where {T,N}
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs::TracedRArray{T,N}, rhs) where {T,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{T,N}) where {T,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end
    end
end

for (jlop, hloop) in ((:(Base.:*), :multiply), (:(Base.:/), :divide), (:(Base.:^), :power))
    @eval begin
        function $jlop(lhs::TracedRArray{T,0}, rhs::TracedRArray{T2,0}) where {T,T2}
            commonTy = TracedRArray{Base.promote_type(T, T2),0}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs::TracedRArray{T,0}, rhs::TracedRArray{T,0}) where {T}
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs::TracedRArray{T,0}, rhs) where {T}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{T,0}) where {T}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end
    end
end

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::Reactant.TracedRArray{T,(),0}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
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
    @eval begin
        function $jlop(lhs::TracedRArray{T,N}) where {T,N}
            return TracedRArray{T,N}(
                (), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1)
            )
        end
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
        function elem_apply(
            ::typeof($jlop), lhs::TracedRArray{T,N}, rhs::TracedRArray{T,N}
        ) where {T,N}
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(
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

        function elem_apply(::typeof($jlop), lhs::TracedRArray{T,N}, rhs) where {T,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(
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

        function elem_apply(::typeof($jlop), lhs, rhs::TracedRArray{T,N}) where {T,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(
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
    end
end

function Base.:*(lhs::TracedRArray{T,2}, rhs::TracedRArray{T,2}) where {T}
    lhsty = MLIR.IR.type(lhs.mlir_data)
    rhsty = MLIR.IR.type(rhs.mlir_data)
    resty = MLIR.IR.TensorType((size(lhs, 1), size(rhs, 2)), eltype(lhsty))
    dot_dimension_numbers = MLIR.API.stablehloDotDimensionNumbersGet(
        MLIR.IR.context(), 0, [], 0, [], 1, [1], 1, [0]
    )
    prec = MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), "DEFAULT")
    )
    precar = MLIR.IR.Attribute([prec, prec])
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dot_general(
            lhs.mlir_data,
            rhs.mlir_data;
            result_0=resty,
            dot_dimension_numbers=dot_dimension_numbers,
            precision_config=precar,
        ),
        1,
    )
    return TracedRArray{T,2}((), res, (size(lhs, 1), size(rhs, 2)))
end

function Base.show(io::IO, X::TracedRArray{T,N}) where {T,N}
    print(io, "TracedRArray{", T, ",", N, "N}(", X.paths, ", ")
    return print(io, X.mlir_data, ")")
end

@inline function Enzyme.Compiler.active_reg_inner(
    ::Type{TracedRArray{T,N}},
    seen::ST,
    world::Union{Nothing,UInt},
    ::Val{justActive}=Val(false),
    ::Val{UnionSret}=Val(false),
)::Enzyme.Compiler.ActivityState where {ST,T,N,justActive,UnionSret}
    if Enzyme.Compiler.active_reg_inner(T, seen, world, Val(justActive), Val(UnionSret)) ==
        Enzyme.Compiler.AnyState
        return Enzyme.Compiler.AnyState
    else
        return Enzyme.Compiler.DupState
    end
end
