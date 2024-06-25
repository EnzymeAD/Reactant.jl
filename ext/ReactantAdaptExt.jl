module ReactantAdaptExt

using Reactant: ConcreteRArray
using Adapt

Adapt.adapt_storage(::Type{T}, x::AbstractArray) where {T<:ConcreteRArray} = T(x)

end
