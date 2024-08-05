module ReactantStatisticsExt

using Reactant: TracedRArray
using Statistics: Statistics

function Statistics.mean(A::TracedRArray{T,Shape,N}; dims=:) where {T,Shape,N}
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)
    return mapreduce(identity, +, A; dims) / denom
end

function Statistics.var(
    A::TracedRArray{T,Shape,N}; dims=:, mean=nothing, corrected=true
) where {T,Shape,N}
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = (dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end

end
