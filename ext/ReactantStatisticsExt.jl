module ReactantStatisticsExt

using Reactant: TracedRArray
using Statistics: Statistics

function Statistics.mean(A::TracedRArray{T,N}; dims=:) where {T,N}
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)
    return mapreduce(identity, +, A; dims) / denom
end

function Statistics.var(
    A::TracedRArray{T,N}; dims=:, mean=nothing, corrected=true
) where {T,N}
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = (dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end

end
