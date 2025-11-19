struct BatchedSVD{T,Tr,M<:AbstractArray,C<:AbstractArray} <: Factorization{T}
    U::M
    S::C
    Vt::M

    function BatchedSVD{T,Tr,M,C}(U::M, S::C, Vt::M) where {T,Tr,M,C}
        @assert ndims(S) == ndims(U) - 1
        return new{T,Tr,M,C}(U, S, Vt)
    end
end
