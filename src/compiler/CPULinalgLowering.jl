# Lower `stablehlo.cholesky` / `mhlo.cholesky` to `enzymexla.lapack.potrf` on the
# CPU backend after Enzyme differentiation so CPU Cholesky factorizations call
# LAPACK `?potrf_` instead of XLA's unblocked/loop Cholesky expander (#3411).

_cholesky_nan_val(::Type{Float32}) = NaN32
_cholesky_nan_val(::Type{Float64}) = NaN
_cholesky_nan_val(::Type{ComplexF32}) = ComplexF32(NaN32, NaN32)
_cholesky_nan_val(::Type{ComplexF64}) = ComplexF64(NaN, NaN)

function _emit_2d_cpu_potrf(
    ctx::MLIR.IR.Context,
    loc::MLIR.IR.Location,
    input_val::MLIR.IR.Value,
    mat_ty::MLIR.IR.Type,
    n::Int64,
    ::Type{T},
    is_lower::Bool,
)::MLIR.IR.Value where {T}
    uplo_enum = if is_lower
        MLIR.API.ENZYMEXLA_LAPACK_UPLO_LOWER
    else
        MLIR.API.ENZYMEXLA_LAPACK_UPLO_UPPER
    end
    uplo_attr = MLIR.IR.Attribute(MLIR.API.enzymexlaLapackUploAttrGet(ctx, uplo_enum))
    i32_ty = MLIR.IR.Type(Int32)
    i64_ty = MLIR.IR.Type(Int64)
    bool_ty = MLIR.IR.Type(Bool)
    el_mlir_ty = MLIR.IR.Type(T)

    info_ty = MLIR.IR.TensorType(Int64[], i32_ty)
    potrf_op = MLIR.Dialects.enzymexla.lapack_potrf(
        input_val; output=mat_ty, info=info_ty, uplo=uplo_attr, location=loc
    )
    factor_raw = MLIR.IR.result(potrf_op, 1)
    info_val = MLIR.IR.result(potrf_op, 2)

    # Zero the untouched opposite triangle to match `stablehlo.cholesky`.
    idx_ty = MLIR.IR.TensorType(Int64[n, n], i64_ty)
    row_op = MLIR.Dialects.stablehlo.iota(;
        output=idx_ty, iota_dimension=MLIR.IR.Attribute(Int64(0)), location=loc
    )
    col_op = MLIR.Dialects.stablehlo.iota(;
        output=idx_ty, iota_dimension=MLIR.IR.Attribute(Int64(1)), location=loc
    )
    tri_mask_ty = MLIR.IR.TensorType(Int64[n, n], bool_ty)
    cmp_dir = is_lower ? "GE" : "LE"
    tri_cmp = MLIR.Dialects.stablehlo.compare(
        MLIR.IR.result(row_op, 1),
        MLIR.IR.result(col_op, 1);
        result_0=tri_mask_ty,
        comparison_direction=MLIR.IR.Attribute(
            MLIR.API.stablehloComparisonDirectionAttrGet(ctx, cmp_dir)
        ),
        location=loc,
    )

    empty_dims = MLIR.IR.DenseArrayAttribute(Int64[])
    scalar_ty = MLIR.IR.TensorType(Int64[], el_mlir_ty)
    zero_scalar = MLIR.Dialects.stablehlo.constant(;
        output=scalar_ty, value=MLIR.IR.DenseElementsAttribute(fill(zero(T))), location=loc
    )
    zero_mat = MLIR.Dialects.stablehlo.broadcast_in_dim(
        MLIR.IR.result(zero_scalar, 1);
        result_0=mat_ty,
        broadcast_dimensions=empty_dims,
        location=loc,
    )
    masked_op = MLIR.Dialects.stablehlo.select(
        MLIR.IR.result(tri_cmp, 1),
        factor_raw,
        MLIR.IR.result(zero_mat, 1);
        result=mat_ty,
        location=loc,
    )

    # Fill with NaN when `info != 0` (non-positive-definite) to match `stablehlo.cholesky`.
    zero_i32 = MLIR.Dialects.stablehlo.constant(;
        output=info_ty, value=MLIR.IR.DenseElementsAttribute(fill(Int32(0))), location=loc
    )
    bool_scalar_ty = MLIR.IR.TensorType(Int64[], bool_ty)
    ok_cmp = MLIR.Dialects.stablehlo.compare(
        info_val,
        MLIR.IR.result(zero_i32, 1);
        result_0=bool_scalar_ty,
        comparison_direction=MLIR.IR.Attribute(
            MLIR.API.stablehloComparisonDirectionAttrGet(ctx, "EQ")
        ),
        location=loc,
    )
    ok_bcast = MLIR.Dialects.stablehlo.broadcast_in_dim(
        MLIR.IR.result(ok_cmp, 1);
        result_0=tri_mask_ty,
        broadcast_dimensions=empty_dims,
        location=loc,
    )
    nan_scalar = MLIR.Dialects.stablehlo.constant(;
        output=scalar_ty,
        value=MLIR.IR.DenseElementsAttribute(fill(_cholesky_nan_val(T))),
        location=loc,
    )
    nan_mat = MLIR.Dialects.stablehlo.broadcast_in_dim(
        MLIR.IR.result(nan_scalar, 1);
        result_0=mat_ty,
        broadcast_dimensions=empty_dims,
        location=loc,
    )
    final_op = MLIR.Dialects.stablehlo.select(
        MLIR.IR.result(ok_bcast, 1),
        MLIR.IR.result(masked_op, 1),
        MLIR.IR.result(nan_mat, 1);
        result=mat_ty,
        location=loc,
    )
    return MLIR.IR.result(final_op, 1)
end

function _emit_cpu_cholesky_replacement(
    ctx::MLIR.IR.Context,
    loc::MLIR.IR.Location,
    input_val::MLIR.IR.Value,
    orig_ty::MLIR.IR.Type,
    shape::Vector{Int64},
    ::Type{T},
    is_lower::Bool,
)::MLIR.IR.Value where {T}
    rank = length(shape)
    n = shape[end]
    el_mlir_ty = MLIR.IR.Type(T)
    mat2d_ty = MLIR.IR.TensorType(Int64[n, n], el_mlir_ty)
    if rank == 2
        return _emit_2d_cpu_potrf(ctx, loc, input_val, mat2d_ty, n, T, is_lower)
    end

    batch_count = Int64(1)
    for d in 1:(rank - 2)
        batch_count *= shape[d]
    end

    if batch_count == 1
        in2d = MLIR.Dialects.stablehlo.reshape(input_val; result_0=mat2d_ty, location=loc)
        out2d = _emit_2d_cpu_potrf(
            ctx, loc, MLIR.IR.result(in2d, 1), mat2d_ty, n, T, is_lower
        )
        out_nd = MLIR.Dialects.stablehlo.reshape(out2d; result_0=orig_ty, location=loc)
        return MLIR.IR.result(out_nd, 1)
    end

    i64_ty = MLIR.IR.Type(Int64)
    bool_ty = MLIR.IR.Type(Bool)

    flat_ty = MLIR.IR.TensorType(Int64[batch_count, n, n], el_mlir_ty)
    slice3d_ty = MLIR.IR.TensorType(Int64[1, n, n], el_mlir_ty)
    flat_in_op = MLIR.Dialects.stablehlo.reshape(input_val; result_0=flat_ty, location=loc)
    flat_in = MLIR.IR.result(flat_in_op, 1)

    idx_scalar_ty = MLIR.IR.TensorType(Int64[], i64_ty)
    bool_scalar_ty = MLIR.IR.TensorType(Int64[], bool_ty)
    zero_idx_op = MLIR.Dialects.stablehlo.constant(;
        output=idx_scalar_ty,
        value=MLIR.IR.DenseElementsAttribute(fill(Int64(0))),
        location=loc,
    )
    one_idx_op = MLIR.Dialects.stablehlo.constant(;
        output=idx_scalar_ty,
        value=MLIR.IR.DenseElementsAttribute(fill(Int64(1))),
        location=loc,
    )
    bound_idx_op = MLIR.Dialects.stablehlo.constant(;
        output=idx_scalar_ty,
        value=MLIR.IR.DenseElementsAttribute(fill(batch_count)),
        location=loc,
    )

    scalar_ty = MLIR.IR.TensorType(Int64[], el_mlir_ty)
    zero_el_op = MLIR.Dialects.stablehlo.constant(;
        output=scalar_ty, value=MLIR.IR.DenseElementsAttribute(fill(zero(T))), location=loc
    )
    init_buf_op = MLIR.Dialects.stablehlo.broadcast_in_dim(
        MLIR.IR.result(zero_el_op, 1);
        result_0=flat_ty,
        broadcast_dimensions=MLIR.IR.DenseArrayAttribute(Int64[]),
        location=loc,
    )

    cond_reg = MLIR.IR.Region()
    cond_blk = MLIR.IR.Block(
        MLIR.IR.Type[idx_scalar_ty, flat_ty], MLIR.IR.Location[loc, loc]
    )
    push!(cond_reg, cond_blk)
    MLIR.IR.activate(cond_blk)
    try
        cmp_op = MLIR.Dialects.stablehlo.compare(
            MLIR.IR.argument(cond_blk, 1),
            MLIR.IR.result(bound_idx_op, 1);
            result_0=bool_scalar_ty,
            comparison_direction=MLIR.IR.Attribute(
                MLIR.API.stablehloComparisonDirectionAttrGet(ctx, "LT")
            ),
            location=loc,
        )
        MLIR.Dialects.stablehlo.return_(
            MLIR.IR.Value[MLIR.IR.result(cmp_op, 1)]; location=loc
        )
    finally
        MLIR.IR.deactivate(cond_blk)
    end

    body_reg = MLIR.IR.Region()
    body_blk = MLIR.IR.Block(
        MLIR.IR.Type[idx_scalar_ty, flat_ty], MLIR.IR.Location[loc, loc]
    )
    push!(body_reg, body_blk)
    MLIR.IR.activate(body_blk)
    try
        iv = MLIR.IR.argument(body_blk, 1)
        acc = MLIR.IR.argument(body_blk, 2)
        z0 = MLIR.IR.result(zero_idx_op, 1)
        sl_op = MLIR.Dialects.stablehlo.dynamic_slice(
            flat_in,
            MLIR.IR.Value[iv, z0, z0];
            result=slice3d_ty,
            slice_sizes=MLIR.IR.DenseArrayAttribute(Int64[1, n, n]),
            location=loc,
        )
        mat_op = MLIR.Dialects.stablehlo.reshape(
            MLIR.IR.result(sl_op, 1); result_0=mat2d_ty, location=loc
        )
        fac2d = _emit_2d_cpu_potrf(
            ctx, loc, MLIR.IR.result(mat_op, 1), mat2d_ty, n, T, is_lower
        )
        fac3d_op = MLIR.Dialects.stablehlo.reshape(fac2d; result_0=slice3d_ty, location=loc)
        dus_op = MLIR.Dialects.stablehlo.dynamic_update_slice(
            acc,
            MLIR.IR.result(fac3d_op, 1),
            MLIR.IR.Value[iv, z0, z0];
            result=flat_ty,
            location=loc,
        )
        next_iv_op = MLIR.Dialects.stablehlo.add(
            iv, MLIR.IR.result(one_idx_op, 1); result=idx_scalar_ty, location=loc
        )
        MLIR.Dialects.stablehlo.return_(
            MLIR.IR.Value[MLIR.IR.result(next_iv_op, 1), MLIR.IR.result(dus_op, 1)];
            location=loc,
        )
    finally
        MLIR.IR.deactivate(body_blk)
    end

    while_op = MLIR.Dialects.stablehlo.while_(
        MLIR.IR.Value[MLIR.IR.result(zero_idx_op, 1), MLIR.IR.result(init_buf_op, 1)];
        result_0=MLIR.IR.Type[idx_scalar_ty, flat_ty],
        cond=cond_reg,
        body=body_reg,
        location=loc,
    )
    out_op = MLIR.Dialects.stablehlo.reshape(
        MLIR.IR.result(while_op, 2); result_0=orig_ty, location=loc
    )
    return MLIR.IR.result(out_op, 1)
end

function _try_rewrite_cholesky_op!(chol_op::MLIR.IR.Operation)::Bool
    orig_ty = MLIR.IR.type(MLIR.IR.result(chol_op, 1))
    MLIR.IR.istensor(orig_ty) || return false
    nd = MLIR.IR.ndims(orig_ty)
    nd >= 2 || return false

    shape = Int64[]
    for d in 1:nd
        sz = size(orig_ty, d)
        (MLIR.IR.isdynsize(sz) || sz <= 0) && return false
        push!(shape, Int64(sz))
    end
    shape[end - 1] == shape[end] || return false

    el_ty = eltype(orig_ty)
    T = if MLIR.API.mlirTypeIsAF32(el_ty)
        Float32
    elseif MLIR.API.mlirTypeIsAF64(el_ty)
        Float64
    elseif MLIR.IR.iscomplex(el_ty)
        inner_ty = eltype(el_ty)
        if MLIR.API.mlirTypeIsAF32(inner_ty)
            ComplexF32
        elseif MLIR.API.mlirTypeIsAF64(inner_ty)
            ComplexF64
        else
            return false
        end
    else
        return false
    end

    lower_attr = MLIR.IR.getattr(chol_op, "lower")
    is_lower =
        lower_attr !== nothing && MLIR.IR.isbool(lower_attr) ? Bool(lower_attr) : false

    ctx = MLIR.IR.context(chol_op)
    loc = MLIR.IR.location(chol_op)
    input_val = MLIR.IR.operand(chol_op, 1)

    tmp_blk = MLIR.IR.Block()
    new_val = try
        MLIR.IR.activate(tmp_blk)
        local v::MLIR.IR.Value
        try
            v = _emit_cpu_cholesky_replacement(ctx, loc, input_val, orig_ty, shape, T, is_lower)
        finally
            MLIR.IR.deactivate(tmp_blk)
        end
        emitted = MLIR.IR.Operation[]
        for eop in tmp_blk
            push!(emitted, eop)
        end
        for eop in emitted
            MLIR.API.mlirOperationMoveBefore(eop, chol_op)
        end
        v
    finally
        MLIR.IR.dispose(tmp_blk)
    end

    MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.result(chol_op, 1), new_val)
    MLIR.API.mlirOperationRemoveFromParent(chol_op)
    MLIR.IR.dispose(chol_op)
    return true
end

function _lower_cpu_cholesky_in_region!(reg::MLIR.IR.Region)::Bool
    changed = false
    for blk in reg
        chol_ops = MLIR.IR.Operation[]
        for op in blk
            op_name = MLIR.IR.name(op)
            if op_name == "stablehlo.cholesky" || op_name == "mhlo.cholesky"
                push!(chol_ops, op)
            else
                for r in 1:MLIR.IR.nregions(op)
                    if _lower_cpu_cholesky_in_region!(MLIR.IR.region(op, r))
                        changed = true
                    end
                end
            end
        end
        for cop in chol_ops
            if _try_rewrite_cholesky_op!(cop)
                changed = true
            end
        end
    end
    return changed
end

function _lower_cpu_cholesky_to_lapack!(mod::MLIR.IR.Module)::Bool
    changed = false
    for top_op in MLIR.IR.body(mod)
        for r in 1:MLIR.IR.nregions(top_op)
            if _lower_cpu_cholesky_in_region!(MLIR.IR.region(top_op, r))
                changed = true
            end
        end
    end
    return changed
end
