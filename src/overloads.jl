function promote_to(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
    if !(rhs <: Number)
        if ElType != eltype(rhs)
            throw(ArgumentError("Cannot promote $(typeof(rhs)) to $(TracedRArray{ElType,Shape,N}) with different element types"))
        end
        if Shape != size(rhs)
            throw(ArgumentError("Cannot promote to TracedRArray with different shapes"))
        end
    end

    if isa(rhs, TracedRArray)
        if isa(rhs, Number)
            throw(ArgumentError("TODO broadcast"))
        end
        return rhs
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(ElType(rhs)), mlir_type(lhs))
        return TracedRArray{ElType,Shape,N}(nothing, MLIR.IR.stablehlo.constant(attr))
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(lhs), rhs)
    return TracedRArray{ElType,Shape,N}(nothing, MLIR.IR.stablehlo.constant(attr))
end

for (jlop, hloop) in ((:+, :add), (:-, :subtract), (:*, :multiply), (:/, :divide))
    @eval begin
        function $jlop(lhs::TracedRArray{ElType,Shape,N}, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end

        function $jlop(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end

        function $jlop(lhs, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{ElType,Shape,N}((),  MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1))
        end
    end
end

for (jlop, hloop) in ((:(Base.sin), :sine), (:(Base.cos), :cosine), (:(Base.exp), :exp), (:(Base.log), :log), (:(Base.sqrt), :sqrt))
    @eval begin
        function $jlop(lhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}((), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1))
        end
    end
end