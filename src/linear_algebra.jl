function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T1,1}),
    @nospecialize(A::AnyTracedRArray{T2,2}),
    @nospecialize(B::AnyTracedRArray{T3,1}),
    α::Number=true,
    β::Number=false,
) where {T1,T2,T3}
    # TODO: The reshape operations are not getting optimized, we should directly call dot_general
    rC = Ops.reshape(C, length(C), 1)
    LinearAlgebra.mul!(rC, A, reshape(B, :, 1), α, β)
    C.mlir_data = get_mlir_data(vec(rC))
    return C
end

function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T1,2}),
    @nospecialize(A::AnyTracedRArray{T2,2}),
    @nospecialize(B::AnyTracedRArray{T3,1}),
    α::Number=true,
    β::Number=false,
) where {T1,T2,T3}
    LinearAlgebra.mul!(C, A, reshape(B, :, 1), α, β)
    return C
end

function LinearAlgebra.mul!(
    @nospecialize(C::TracedRArray{T1,2}),
    @nospecialize(A::AnyTracedRArray{T2,2}),
    @nospecialize(B::AnyTracedRArray{T3,2}),
    α::Number=true,
    β::Number=false,
) where {T1,T2,T3}
    if size(C) != (size(A, 1), size(B, 2))
        throw(
            DimensionMismatch(
                "C has size $(size(C)), A has size $(size(A)), B has size $(size(B))"
            ),
        )
    end
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    end
    resty = MLIR.IR.TensorType(size(C), MLIR.IR.Type(T1))
    dot_dimension_numbers = MLIR.API.stablehloDotDimensionNumbersGet(
        MLIR.IR.context(), 0, [], 0, [], 1, [1], 1, [0]
    )
    prec = MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), "DEFAULT")
    )
    precar = MLIR.IR.Attribute([prec, prec])
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dot_general(
            get_mlir_data(A),
            get_mlir_data(B);
            result_0=resty,
            dot_dimension_numbers=dot_dimension_numbers,
            precision_config=precar,
        ),
        1,
    )
    if iszero(β)
        if isone(α)
            C.mlir_data = res
        else
            C.mlir_data = MLIR.IR.result(
                MLIR.Dialects.stablehlo.multiply(
                    res, broadcast_to_size(T1(α), size(C)).mlir_data
                ),
                1,
            )
        end
    else
        α_res = MLIR.IR.result(
            MLIR.Dialects.stablehlo.multiply(
                res, broadcast_to_size(T1(α), size(C)).mlir_data
            ),
            1,
        )
        β_C = MLIR.IR.result(
            MLIR.Dialects.stablehlo.multiply(
                C.mlir_data, broadcast_to_size(T1(β), size(C)).mlir_data
            ),
            1,
        )
        C.mlir_data = MLIR.IR.result(MLIR.Dialects.stablehlo.add(α_res, β_C), 1)
    end
    return C
end

function LinearAlgebra.triu!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = Ops.iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = Ops.subtract(
        Ops.iota(Int64, [size(X)...]; iota_dimension=2), broadcast_to_size(k, size(X))
    )
    idxs = Ops.compare(iota_1, iota_2; comparison_direction="LE")
    X.mlir_data = Ops.select(idxs, X, zero(X)).mlir_data
    return X
end

function LinearAlgebra.tril!(@nospecialize(X::TracedRArray{T,2}), k::Integer) where {T}
    iota_1 = Ops.iota(Int64, [size(X)...]; iota_dimension=1)
    iota_2 = Ops.subtract(
        Ops.iota(Int64, [size(X)...]; iota_dimension=2), broadcast_to_size(k, size(X))
    )
    idxs = Ops.compare(iota_1, iota_2; comparison_direction="GE")
    X.mlir_data = Ops.select(idxs, X, zero(X)).mlir_data
    return X
end

# LinearAlgebra defines norm with some conditionals which cannot be traced directly
function LinearAlgebra.norm(x::TracedRArray{T,N}, p::Real=2) where {T,N}
    isinf(p) && return maximum(abs, x)
    return mapreduce(Base.Fix2(^, p), +, x)^(1 / p)
end
