function TracedUtils.materialize_traced_array(
    x::Transpose{TracedRNumber{T},TracedRArray{T,N}}
) where {T,N}
    px = parent(x)
    A = ndims(px) == 1 ? reshape(px, :, 1) : px
    return permutedims(A, (2, 1))
end

function TracedUtils.materialize_traced_array(
    x::Transpose{TracedRNumber{T},<:WrappedTracedRArray{T,N}}
) where {T,N}
    return materialize_traced_array(transpose(materialize_traced_array(parent(x))))
end

function TracedUtils.materialize_traced_array(
    x::Adjoint{TracedRNumber{T},TracedRArray{T,N}}
) where {T,N}
    return conj(materialize_traced_array(transpose(parent(x))))
end

function TracedUtils.materialize_traced_array(
    x::Adjoint{TracedRNumber{T},<:WrappedTracedRArray{T,N}}
) where {T,N}
    return materialize_traced_array(adjoint(materialize_traced_array(parent(x))))
end

function TracedUtils.materialize_traced_array(
    x::Diagonal{TracedRNumber{T},TracedRArray{T,1}}
) where {T}
    return diagm(parent(x))
end

function TracedUtils.materialize_traced_array(
    x::Diagonal{TracedRNumber{T},WrappedTracedRArray{T,1}}
) where {T}
    return diagm(materialize_traced_array(parent(x)))
end

function TracedUtils.materialize_traced_array(
    x::Tridiagonal{TracedRNumber{T},TracedRArray{T,1}}
) where {T}
    return diagm(-1 => x.dl, 0 => x.d, 1 => x.du)
end

for (AT, comp) in ((:LowerTriangular, "GE"), (:UpperTriangular, "LE"))
    uAT = Symbol(:Unit, AT)
    @eval begin
        function TracedUtils.materialize_traced_array(
            x::$(AT){TracedRNumber{T},TracedRArray{T,2}}
        ) where {T}
            m, n = size(x)
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(comp))
            return Ops.select(indicator, parent(x), zero(parent(x)))
        end

        function TracedUtils.materialize_traced_array(
            x::$(uAT){TracedRNumber{T},TracedRArray{T,2}}
        ) where {T}
            m, n = size(x)
            row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
            col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
            nondiag_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="NE")
            x = materialize_traced_array($(AT)(parent(x)))
            return Ops.select(nondiag_indicator, x, one.(x))
        end
    end
end

function TracedUtils.materialize_traced_array(
    x::Symmetric{TracedRNumber{T},TracedRArray{T,2}}
) where {T}
    m, n = size(x)
    row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
    col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
    if x.uplo == 'L'
        indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="GT")
        x_lt = Ops.select(indicator, parent(x), zero(parent(x)))
        x_ltd = materialize_traced_array(LowerTriangular(parent(x)))
        return Ops.add(x_lt, Ops.transpose(x_ltd, [2, 1]))
    else
        indicator = Ops.compare(row_idxs, col_idxs; comparison_direction="LT")
        x_ut = Ops.select(indicator, parent(x), zero(parent(x)))
        x_utd = materialize_traced_array(UpperTriangular(parent(x)))
        return Ops.add(Ops.transpose(x_utd, [2, 1]), x_ut)
    end
end

function TracedUtils.set_mlir_data!(
    x::Transpose{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    px.mlir_data = (
        if ndims(px) == 1
            Ops.reshape(tdata, length(tdata))
        else
            Ops.transpose(tdata, [2, 1])
        end
    ).mlir_data
    return x
end

function TracedUtils.set_mlir_data!(
    x::Adjoint{TracedRNumber{T},TracedRArray{T,N}}, data
) where {T,N}
    tdata = TracedRArray{T}(data)
    px = parent(x)
    transposed_data =
        ndims(px) == 1 ? Ops.reshape(tdata, length(tdata)) : Ops.transpose(tdata, [2, 1])
    px.mlir_data = (T <: Real ? transposed_data : Ops.conj(transposed_data)).mlir_data
    return x
end

function TracedUtils.set_mlir_data!(
    x::Diagonal{TracedRNumber{T},TracedRArray{T,1}}, data
) where {T}
    parent(x).mlir_data = diag(TracedRArray{T}(data)).mlir_data
    return x
end

for (AT, dcomp, ocomp) in (
    (:LowerTriangular, "GE", "LT"),
    (:UnitLowerTriangular, "GT", "LE"),
    (:UpperTriangular, "LE", "GT"),
    (:UnitUpperTriangular, "LT", "GE"),
)
    @eval function TracedUtils.set_mlir_data!(
        x::$(AT){TracedRNumber{T},TracedRArray{T,2}}, data
    ) where {T}
        tdata = TracedRArray{T}(data)
        z = zero(tdata)
        m, n = size(x)
        row_idxs = Ops.iota(Int, [m, n]; iota_dimension=1)
        col_idxs = Ops.iota(Int, [m, n]; iota_dimension=2)
        data_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(dcomp))
        original_indicator = Ops.compare(row_idxs, col_idxs; comparison_direction=$(ocomp))
        res = Ops.add(
            Ops.select(data_indicator, tdata, z), Ops.select(original_indicator, x.data, z)
        )
        set_mlir_data!(x.data, res.mlir_data)
        return x
    end
end

function TracedUtils.set_mlir_data!(
    x::Symmetric{TracedRNumber{T},TracedRArray{T,2}}, data
) where {T}
    if x.uplo == 'L'
        set_mlir_data!(LowerTriangular(parent(x)), data)
    else
        set_mlir_data!(UpperTriangular(parent(x)), data)
    end
    return x
end

function TracedUtils.set_mlir_data!(
    x::Tridiagonal{TracedRNumber{T},TracedRArray{T,1}}, data
) where {T}
    tdata = TracedRArray{T}(data)
    set_mlir_data!(x.dl, diag(tdata, -1).mlir_data)
    set_mlir_data!(x.d, diag(tdata, 0).mlir_data)
    set_mlir_data!(x.du, diag(tdata, 1).mlir_data)
    return x
end
