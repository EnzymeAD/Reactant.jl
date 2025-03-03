function LAPACK.chkfinite(x::TracedRArray{T, 2}) where {T}
    res = all(Ops.is_finite(x))
    # TODO: runtime error in MLIR
    # not_finite = !res
    # @trace if not_finite
    #     error("Matrix contains Infs or NaNs")
    # end
    return res
end
