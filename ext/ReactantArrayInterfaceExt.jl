module ReactantArrayInterfaceExt

using ArrayInterface: ArrayInterface
using Reactant:
    Reactant, RArray, ConcreteRArray, ConcreteRNumber, TracedRNumber, TracedRArray, Ops

ArrayInterface.can_setindex(::Type{<:RArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:RArray}) = false

for aType in
    (AbstractArray{<:ConcreteRNumber}, AbstractArray{<:TracedRNumber}, TracedRArray)
    @eval ArrayInterface.aos_to_soa(x::$aType) = Reactant.aos_to_soa(x)
end

end
