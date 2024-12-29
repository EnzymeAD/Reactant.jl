module ReactantArrayInterfaceExt

using ArrayInterface: ArrayInterface
using Reactant:
    Reactant, RArray, ConcreteRArray, ConcreteRNumber, TracedRNumber, TracedRArray, Ops

ArrayInterface.can_setindex(::Type{<:RArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:RArray}) = false

function ArrayInterface.aos_to_soa(x::AbstractArray{<:ConcreteRNumber{T}}) where {T}
    x_c = ConcreteRArray(zeros(T, size(x)))
    x_c .= x
    return x_c
end

ArrayInterface.aos_to_soa(x::TracedRArray) = x
function ArrayInterface.aos_to_soa(x::AbstractArray{<:TracedRNumber{T}}) where {T}
    return Ops.reshape(vcat(x...), size(x)...)
end

end
