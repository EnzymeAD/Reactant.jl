# Conversion between SparseArrays types and the opaque `Reactant.CSRMatrix`.

function Reactant.CSRMatrix(A::SparseMatrixCSC{T,Ti}) where {T,Ti}
    At = copy(transpose(A)) # CSC of Aᵀ is the CSR representation of A
    return Reactant.CSRMatrix{T,Ti,Vector{T},Vector{Ti}}(
        size(A, 1), size(A, 2), At.colptr, At.rowval, At.nzval
    )
end

function Reactant.to_rarray(A::SparseMatrixCSC; kwargs...)
    return Reactant.to_rarray(Reactant.CSRMatrix(A); kwargs...)
end

SparseArrays.nnz(A::Reactant.CSRMatrix) = length(A.colind)

function SparseArrays.SparseMatrixCSC(A::Reactant.CSRMatrix{T,Ti}) where {T,Ti}
    # The CSR buffers of A are the CSC representation of Aᵀ
    At = SparseMatrixCSC{T,Ti}(
        A.n, A.m, Vector{Ti}(A.rowptr), Vector{Ti}(A.colind), Vector{T}(A.nzval)
    )
    return copy(transpose(At))
end
