struct BatchedQR{T,M<:AbstractArray{T}} <: BatchedFactorization{T}
    Q::M
    R::M

    function BatchedQR{T,M}(Q::M, R::M) where {T,M}
        @assert size(Q, 2) == size(R, 1)
        @assert size(Q)[3:end] == size(R)[3:end]
        return new{T,M}(Q, R)
    end
end

BatchedQR(Q::M, R::M) where {M} = BatchedQR{eltype(Q),M}(Q, R)    

Base.size(qr::BatchedQR) = (size(qr.Q, 1), size(qr.R, 2), size(qr.Q)[3:end]...)
Base.size(qr::BatchedQR, i::Integer) = i == 2 ? size(qr.R, 2) : size(qr.Q, i)

Base.iterate(F::BatchedQR) = (F.Q, Val(:R))
Base.iterate(F::BatchedQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(::BatchedQR, ::Val{:done}) = nothing

function overloaded_qr(A::AnyTracedRArray, ::NoPivot; blocksize=nothing)
    @assert isnothing(blocksize) "Block size is not yet supported for QR factorization"

    # Batching here is in the last dimensions. `Ops.qr` expects the last dimensions
    permdims = vcat(collect(Int64, 3:ndims(A)), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)

    Q, R = @opcall qr(A)

    # Permute back to the original dimensions
    Q = @opcall transpose(Q, invperm(permdims))
    R = @opcall transpose(R, invperm(permdims))

    return BatchedQR(Q, R)
end

function overloaded_qr(A::AbstractArray; kwargs...)
    return overloaded_qr(Reactant.promote_to(TracedRArray, A); kwargs...)
end

# TODO ldiv!(A::BatchedQR, B::AbstractArray)
