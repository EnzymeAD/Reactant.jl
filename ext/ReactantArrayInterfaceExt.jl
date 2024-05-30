module ReactantArrayInterfaceExt

using ArrayInterface: ArrayInterface
using Reactant: RArray

ArrayInterface.can_setindex(::Type{<:RArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:RArray}) = false

end
