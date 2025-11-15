struct GeneralizedCholesky{T,S<:AbstractArray,I<:Union{AbstractArray,Number}} <:
       GeneralizedFactorization{T}
    factors::S
    uplo::Char
    info::I
end

Base.size(c::GeneralizedCholesky) = size(c.factors)
Base.ndims(c::GeneralizedCholesky) = ndims(c.factors)

function overloaded_cholesky(A::AbstractArray, ::NoPivot; check::Bool=false)
    return overloaded_cholesky(Reactant.promote_to(TracedRArray, A), NoPivot(); check)
end

function overloaded_cholesky(
    A::AnyTracedRArray{T,N}, ::NoPivot; check::Bool=false
) where {T,N}
    # TODO: dont ignore check
    # move the batching dims to the front
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)

    factors = @opcall cholesky(A; lower=false)
    factors = @opcall transpose(factors, invperm(permdims))

    # stablehlo doesn't return the info
    info = materialize_traced_array(
        dropdims(
            Reactant.CallWithReactant(mapreduce)(
                isfinite, &, UpperTriangular(factors); dims=1:2
            );
            dims=(1, 2),
        ),
    )

    return GeneralizedCholesky{T,typeof(factors),typeof(info)}(factors, 'U', info)
end
