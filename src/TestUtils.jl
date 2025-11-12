module TestUtils

using ..Reactant: Reactant, TracedRArray
using ReactantCore: ReactantCore
using LinearAlgebra: LinearAlgebra

function construct_test_array(::Type{T}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(flat_vector, dims...)
end

function construct_test_array(::Type{Complex{T}}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(complex.(flat_vector, flat_vector), dims...)
end

function construct_test_array(::Type{T}, dims::Int...) where {T}
    return reshape(collect(T, 1:prod(dims)), dims...)
end

# https://github.com/JuliaDiff/FiniteDiff.jl/blob/3a8c3d8d87e59de78e2831787a3f54b12b7c2075/src/epsilons.jl#L133
function default_epslion(::Val{fdtype}, ::Type{T}) where {fdtype,T}
    if fdtype == :forward
        return sqrt(eps(real(T)))
    elseif fdtype == :central
        return cbrt(eps(real(T)))
    elseif fdtype == :hcentral
        return eps(T)^(T(1 / 4))
    else
        return one(real(T))
    end
end

function finite_difference_gradient(
    f, x::AbstractArray{T}; epsilon=default_epslion(Val(:central), T)
) where {T}
    onehot_matrix = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(T),2},
        LinearAlgebra.Diagonal(fill(epsilon, length(x))),
    )
    perturbation = reshape(onehot_matrix, size(x)..., length(x))
    f_input = cat(x .+ perturbation, x .- perturbation; dims=ndims(x) + 1)

    f_evaluated = mapslices(f, f_input; dims=ntuple(identity, ndims(x)))
    return ReactantCore.materialize_traced_array(
        reshape(
            (f_evaluated[1:length(x)] - f_evaluated[(length(x) + 1):end]) ./ (2 * epsilon),
            size(x),
        ),
    )
end

end
