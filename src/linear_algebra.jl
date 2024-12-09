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

    tmp = Ops.dot_general(
        T1.(materialize_traced_array(A)),
        T1.(materialize_traced_array(B));
        contracting_dimensions=([2], [1]),
    )

    res = if iszero(β)
        isone(α) ? tmp : Ops.multiply(tmp, broadcast_to_size(T1(α), size(C)))
    else
        α_res = Ops.multiply(tmp, broadcast_to_size(T1(α), size(C)))
        β_C = Ops.multiply(C, broadcast_to_size(T1(β), size(C)))
        Ops.add(α_res, β_C)
    end
    set_mlir_data!(C, get_mlir_data(res))
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
