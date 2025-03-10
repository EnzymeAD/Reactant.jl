module ReactantArrayInterfaceExt

using ArrayInterface: ArrayInterface
using Reactant:
    Reactant,
    RArray,
    AbstractConcreteNumber,
    TracedRNumber,
    TracedRArray,
    AnyTracedRArray,
    Ops

ArrayInterface.can_setindex(::Type{<:RArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:RArray}) = false

for aType in (
    AbstractArray{<:AbstractConcreteNumber}, AbstractArray{<:TracedRNumber}, AnyTracedRArray
)
    @eval ArrayInterface.aos_to_soa(x::$aType) = Reactant.aos_to_soa(x)
end

end
