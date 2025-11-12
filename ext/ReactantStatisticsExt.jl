module ReactantStatisticsExt

using Reactant: AnyTracedRArray
using Statistics: Statistics

function Statistics._mean(f::F, A::AnyTracedRArray{T,N}, dims) where {F,T,N}
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)
    return mapreduce(f, +, A; dims) / denom
end

function Statistics._var(
    A::AnyTracedRArray{T,N}, corrected::Bool, mean, ::Colon
) where {T,N}
    mean === nothing && (mean = Statistics.mean(A))
    denom = length(A) - corrected
    return mapreduce(abs2, +, A .- mean; dims=:) / denom
end

function Statistics._var(A::AnyTracedRArray{T,N}, corrected::Bool, mean, dims) where {T,N}
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = prod(Base.Fix1(size, A), dims) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end

end
