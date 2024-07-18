mutable struct TracedRArray{ElType,N} <: RArray{ElType,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{ElType,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {ElType,N}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape
        end
        return new{ElType,N}(paths, mlir_data, shape)
    end
end

Base.size(x::TracedRArray) = x.shape

function Base.similar(x::TracedRArray{T,N}, ::Type{T2}) where {T,N,T2}
    return TracedRArray{T2,N}((), nothing, size(x))
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
        function $jlop(lhs::TracedRArray{ElType,N}) where {ElType,N}
            return TracedRArray{ElType,N}(
                (), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1)
            )
        end
    end
end

for (jlop, hloop, hlocomp, RT) in (
    (:(Base.:(==)), :compare, "EQ", :ElType),
    (:(Base.:(!=)), :compare, "NE", :ElType),
    (:(Base.:(>=)), :compare, "GE", :ElType),
    (:(Base.:(>)), :compare, "GT", :ElType),
    (:(Base.:(<=)), :compare, "LE", :ElType),
    (:(Base.:(<)), :compare, "LT", :ElType),
)
    @eval begin
        function elem_apply(
            ::typeof($jlop),
            lhs::TracedRArray{ElType,Shape,N},
            rhs::TracedRArray{ElType,Shape,N},
        ) where {ElType,Shape,N}
            return TracedRArray{$RT,Shape,N}(
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

        function elem_apply(
            ::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}, rhs
        ) where {ElType,Shape,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,Shape,N}(
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

        function elem_apply(
            ::typeof($jlop), lhs, rhs::TracedRArray{ElType,Shape,N}
        ) where {ElType,Shape,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{$RT,Shape,N}(
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
