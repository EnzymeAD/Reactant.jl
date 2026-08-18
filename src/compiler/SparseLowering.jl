# Lowers the `sparse_tensor.assemble` + `stablehlo.dot_general` pairs emitted by
# `Reactant.sparse_csr_dot` into `stablehlo.custom_call @reactant_csr_matmul`
# operating on the raw CSR buffers. Must run before any pass pipeline: XLA
# cannot consume sparse-encoded tensor types, and the sparse-encoded
# `dot_general` must never reach the verifier.

function _walk_operations!(f, op::MLIR.IR.Operation)
    f(op)
    for region in op, block in region, inner in block
        _walk_operations!(f, inner)
    end
    return nothing
end

function _is_csr_dot(op::MLIR.IR.Operation)
    MLIR.IR.name(op) == "stablehlo.dot_general" || return false
    MLIR.IR.noperands(op) == 2 || return false
    lhs = MLIR.IR.operand(op, 1)
    MLIR.IR.is_op_res(lhs) || return false
    return MLIR.IR.name(MLIR.IR.op_owner(lhs)) == "sparse_tensor.assemble"
end

function _lower_csr_dot!(dot::MLIR.IR.Operation)
    asm = MLIR.IR.op_owner(MLIR.IR.operand(dot, 1))
    rowptr = MLIR.IR.operand(asm, 1)
    colind = MLIR.IR.operand(asm, 2)
    nzval = MLIR.IR.operand(asm, 3)
    rhs = MLIR.IR.operand(dot, 2)

    sparse_type = MLIR.IR.type(MLIR.IR.operand(dot, 1))
    m, n = size(sparse_type, 1), size(sparse_type, 2)
    result_type = MLIR.IR.type(MLIR.IR.result(dot, 1))

    cc = MLIR.Dialects.stablehlo.custom_call(
        MLIR.IR.Value[rowptr, colind, nzval, rhs];
        result_0=MLIR.IR.Type[result_type],
        call_target_name="reactant_csr_matmul",
        api_version=Int32(4),
        has_side_effect=MLIR.IR.Attribute(false),
        backend_config=Dict(
            "m" => MLIR.IR.Attribute(Int64(m)),
            "n" => MLIR.IR.Attribute(Int64(n)),
            "transpose" => MLIR.IR.Attribute(Int64(0)),
            "index_base" => MLIR.IR.Attribute(Int64(1)),
        ),
        operand_layouts=MLIR.IR.Attribute([
            Reactant.Ops._col_major_layout(1),
            Reactant.Ops._col_major_layout(1),
            Reactant.Ops._col_major_layout(1),
            Reactant.Ops._col_major_layout(ndims(MLIR.IR.type(rhs))),
        ]),
        result_layouts=MLIR.IR.Attribute([Reactant.Ops._col_major_layout(ndims(result_type))]),
        location=MLIR.IR.location(dot),
    )

    MLIR.IR.insert_before!(MLIR.IR.block(dot), dot, cc)
    MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.result(dot, 1), MLIR.IR.result(cc, 1))
    MLIR.IR.rmfromparent!(dot)
    MLIR.IR.dispose(dot)
    return asm
end

function lower_sparse_ops!(mod::MLIR.IR.Module)
    # `stablehlo.custom_call` ops created below must stay detached until we
    # insert them next to the `dot_general` they replace.
    @assert !MLIR.IR.has_block()

    has_sparse = false
    dots = MLIR.IR.Operation[]
    for top in collect(MLIR.IR.body(mod))
        _walk_operations!(top) do op
            startswith(MLIR.IR.name(op), "sparse_tensor.") && (has_sparse = true)
            _is_csr_dot(op) && push!(dots, op)
        end
    end
    has_sparse || return mod

    assembles = MLIR.IR.Operation[]
    for dot in dots
        push!(assembles, _lower_csr_dot!(dot))
    end

    seen = Set{Ptr{Cvoid}}()
    for asm in assembles
        ptr = Base.unsafe_convert(MLIR.API.MlirOperation, asm).ptr
        ptr in seen && continue
        push!(seen, ptr)
        if MLIR.IR.first_use(MLIR.IR.result(asm, 1)) === nothing
            MLIR.IR.rmfromparent!(asm)
            MLIR.IR.dispose(asm)
        end
    end

    for top in collect(MLIR.IR.body(mod))
        _walk_operations!(top) do op
            opname = MLIR.IR.name(op)
            if startswith(opname, "sparse_tensor.")
                error(
                    "Unsupported use of the sparse_tensor dialect: `$opname` survived " *
                    "sparse lowering. Only CSR `A * x` / `A * B` products (as emitted " *
                    "by `Reactant.sparse_csr_dot`) can be lowered.",
                )
            end
        end
    end

    return mod
end
