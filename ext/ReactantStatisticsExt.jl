module ReactantStatisticsExt

using Reactant: AnyTracedRArray
using Reactant.TracedUtils: materialize_traced_array
using Statistics: Statistics

function Statistics.mean(A::AnyTracedRArray{T,N}; dims=:) where {T,N}
    A = materialize_traced_array(A)
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)
    return mapreduce(identity, +, A; dims) / denom
end

function Statistics.var(
    A::AnyTracedRArray{T,N}; dims=:, mean=nothing, corrected=true
) where {T,N}
    A = materialize_traced_array(A)
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = (dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end

end
