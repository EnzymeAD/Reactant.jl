module TestUtils

using ..Reactant: Reactant, TracedRArray
using ReactantCore: ReactantCore
using LinearAlgebra: LinearAlgebra

function construct_test_array(::Type{T}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(flat_vector, dims...)
end

function construct_test_array(::Type{T}, dims::Int...) where {T}
    return reshape(collect(T, 1:prod(dims)), dims...)
end

function finite_difference_gradient(f, x::AbstractArray{T}; epsilon=sqrt(eps(T))) where {T}
    onehot_matrix = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(T),2}, LinearAlgebra.I(length(x))
    )
    perturbation = reshape(onehot_matrix .* epsilon, size(x)..., length(x))
    f_input = cat(x .+ perturbation, x .- perturbation; dims=ndims(x) + 1)

    f_evaluated = mapslices(f, f_input; dims=ntuple(identity, ndims(x)))
    return ReactantCore.materialize_traced_array(
        reshape(
            (f_evaluated[1:length(x)] - f_evaluated[(length(x) + 1):end]) ./
            (2 * epsilon),
            size(x),
        ),
    )
end

end
