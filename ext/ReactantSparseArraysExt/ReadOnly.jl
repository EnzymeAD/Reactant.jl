function Base.getindex(
    x::ReadOnly{TracedRNumber{T},N},
    idx::Vararg{Union{Integer,Reactant.TracedRNumber{<:Integer}},N},
) where {T,N}
    return @invoke getindex(x::AbstractArray{TracedRNumber{T},N}, idx...)
end
