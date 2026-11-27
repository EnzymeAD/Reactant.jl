function _gtsv_lapack_eltype(::Type{T}) where {T}
    T <: Union{Float32,Float64,ComplexF32,ComplexF64} && return T
    if T <: Complex
        return Complex{promote_type(float(real(T)), Float32)}
    end
    return promote_type(float(T), Float32)
end

function _tridiag_solve_raw(
    dl::AbstractVector, d::AbstractVector, du::AbstractVector, B::AbstractVecOrMat
)
    m = length(d)
    expected_off = max(m - 1, 0)
    if length(dl) != expected_off || length(du) != expected_off
        throw(
            DimensionMismatch(
                "off-diagonals have lengths ($(length(dl)), $(length(du))), expected $expected_off for main diagonal of length $m",
            ),
        )
    end
    if size(B, 1) != m
        throw(DimensionMismatch("B has $(size(B, 1)) rows, expected $m"))
    end

    Td = promote_type(unwrapped_eltype(dl), unwrapped_eltype(d), unwrapped_eltype(du))
    TB = unwrapped_eltype(B)
    TS = Base.promote_op(\, Td, TB)
    TL = _gtsv_lapack_eltype(TS)

    if m == 0
        return @opcall fill(zero(TS), size(B))
    elseif m == 1
        d_1 = promote_to(TracedRArray{TL,1}, d)
        res = if B isa AbstractVector
            promote_to(TracedRArray{TL,1}, B) ./ d_1
        else
            promote_to(TracedRArray{TL,2}, B) ./ reshape(d_1, 1, 1)
        end
        return TL === TS ? res : promote_to(TracedRArray{TS,ndims(B)}, res)
    end

    dl_v = promote_to(TracedRArray{TL,1}, dl)
    d_v = promote_to(TracedRArray{TL,1}, d)
    du_v = promote_to(TracedRArray{TL,1}, du)
    z1 = @opcall fill(zero(TL), (1,))
    dl_pad = vcat(z1, dl_v)
    du_pad = vcat(du_v, z1)

    res = if B isa AbstractVector
        B_mat = materialize_traced_array(reshape(promote_to(TracedRArray{TL,1}, B), m, 1))
        X_mat, _ = @opcall gtsv(dl_pad, d_v, du_pad, B_mat)
        materialize_traced_array(reshape(X_mat, m))
    else
        B_mat = promote_to(TracedRArray{TL,2}, B)
        X_mat, _ = @opcall gtsv(dl_pad, d_v, du_pad, B_mat)
        X_mat
    end
    return TL === TS ? res : promote_to(TracedRArray{TS,ndims(B)}, res)
end

function Base.:(\)(A::Tridiagonal{<:TracedRNumber}, B::AbstractVecOrMat)
    return _tridiag_solve_raw(A.dl, A.d, A.du, B)
end
function Base.:(\)(A::Tridiagonal, B::Reactant.AnyTracedRVecOrMat)
    return _tridiag_solve_raw(A.dl, A.d, A.du, B)
end
function Base.:(\)(A::Tridiagonal{<:TracedRNumber}, B::Reactant.AnyTracedRVecOrMat)
    return _tridiag_solve_raw(A.dl, A.d, A.du, B)
end

function Base.:(\)(A::SymTridiagonal{<:TracedRNumber}, B::AbstractVecOrMat)
    return _tridiag_solve_raw(A.ev, A.dv, A.ev, B)
end
function Base.:(\)(A::SymTridiagonal, B::Reactant.AnyTracedRVecOrMat)
    return _tridiag_solve_raw(A.ev, A.dv, A.ev, B)
end
function Base.:(\)(A::SymTridiagonal{<:TracedRNumber}, B::Reactant.AnyTracedRVecOrMat)
    return _tridiag_solve_raw(A.ev, A.dv, A.ev, B)
end

function Base.:(\)(A::Transpose{<:TracedRNumber,<:Tridiagonal}, B::AbstractVecOrMat)
    P = parent(A)
    return _tridiag_solve_raw(P.du, P.d, P.dl, B)
end
function Base.:(\)(A::Adjoint{<:TracedRNumber,<:Tridiagonal}, B::AbstractVecOrMat)
    P = parent(A)
    return _tridiag_solve_raw(conj.(P.du), conj.(P.d), conj.(P.dl), B)
end
function Base.:(\)(
    A::Adjoint{<:TracedRNumber,<:Tridiagonal}, B::Adjoint{<:Any,<:AbstractVecOrMat}
)
    P = parent(A)
    return _tridiag_solve_raw(conj.(P.du), conj.(P.d), conj.(P.dl), B)
end

function Base.:(\)(A::Transpose{<:TracedRNumber,<:SymTridiagonal}, B::AbstractVecOrMat)
    P = parent(A)
    return _tridiag_solve_raw(P.ev, P.dv, P.ev, B)
end
function Base.:(\)(A::Adjoint{<:TracedRNumber,<:SymTridiagonal}, B::AbstractVecOrMat)
    P = parent(A)
    return _tridiag_solve_raw(conj.(P.ev), conj.(P.dv), conj.(P.ev), B)
end

function LinearAlgebra.ldiv!(
    A::Tridiagonal{<:TracedRNumber}, B::Reactant.AnyTracedRVecOrMat
)
    copyto!(B, _tridiag_solve_raw(A.dl, A.d, A.du, B))
    return B
end
function LinearAlgebra.ldiv!(A::Tridiagonal, B::Reactant.AnyTracedRVecOrMat)
    copyto!(B, _tridiag_solve_raw(A.dl, A.d, A.du, B))
    return B
end

_apply_symtridiag_shift(dv, shift::TracedRNumber) = dv .+ shift
_apply_symtridiag_shift(dv, shift::Number) = iszero(shift) ? dv : dv .+ shift

function LinearAlgebra.ldiv!(
    A::SymTridiagonal{<:TracedRNumber}, B::AbstractVecOrMat; shift::Number=false
)
    dv = _apply_symtridiag_shift(A.dv, shift)
    copyto!(B, _tridiag_solve_raw(A.ev, dv, A.ev, B))
    return B
end
function LinearAlgebra.ldiv!(
    A::SymTridiagonal, B::Reactant.AnyTracedRVecOrMat; shift::Number=false
)
    dv = _apply_symtridiag_shift(A.dv, shift)
    copyto!(B, _tridiag_solve_raw(A.ev, dv, A.ev, B))
    return B
end
function LinearAlgebra.ldiv!(
    A::SymTridiagonal{<:TracedRNumber}, B::Reactant.AnyTracedRVecOrMat; shift::Number=false
)
    dv = _apply_symtridiag_shift(A.dv, shift)
    copyto!(B, _tridiag_solve_raw(A.ev, dv, A.ev, B))
    return B
end

function LinearAlgebra.ldiv!(
    Y::Reactant.AnyTracedRVecOrMat, A::Tridiagonal, B::AbstractVecOrMat
)
    size(Y) == size(B) ||
        throw(DimensionMismatch("Y has size $(size(Y)), but B has size $(size(B))"))
    copyto!(Y, _tridiag_solve_raw(A.dl, A.d, A.du, B))
    return Y
end
function LinearAlgebra.ldiv!(
    Y::Reactant.AnyTracedRVecOrMat, A::SymTridiagonal, B::AbstractVecOrMat
)
    size(Y) == size(B) ||
        throw(DimensionMismatch("Y has size $(size(Y)), but B has size $(size(B))"))
    copyto!(Y, _tridiag_solve_raw(A.ev, A.dv, A.ev, B))
    return Y
end
