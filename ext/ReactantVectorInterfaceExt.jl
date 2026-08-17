module ReactantVectorInterfaceExt

using Reactant: Reactant, TracedRArray, TracedRNumber, promote_to
using VectorInterface: VectorInterface, Zero, One

Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::Zero) where {T} = promote_to(TT, zero(T))
Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::One) where {T} = promote_to(TT, one(T))

Reactant.promote_to(::Type{TracedRNumber}, v::Zero) = promote_to(TracedRNumber{Float64}, v)
Reactant.promote_to(::Type{TracedRNumber}, v::One) = promote_to(TracedRNumber{Float64}, v)

VectorInterface.scale(x::TracedRNumber, α::Number) = x * α
VectorInterface.scale!!(x::TracedRNumber, α::Number) = VectorInterface.scale!(x, α)
function VectorInterface.scale!(x::TracedRNumber, α::Number)
    y = x * α
    TracedUtils.set_mlir_data!(x, TracedUtils.get_mlir_data(y))
    return x
end

VectorInterface.scale(x::TracedRArray, α::Number) = x * α
VectorInterface.scale!!(x::TracedRArray, α::Number) = VectorInterface.scale!(x, α)
function VectorInterface.scale!(x::TracedRArray, α::Number)
    y = x * α
    TracedUtils.set_mlir_data!(x, TracedUtils.get_mlir_data(y))
    return x
end

end
