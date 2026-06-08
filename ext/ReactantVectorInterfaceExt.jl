module ReactantVectorInterfaceExt

using Rectant
using VectorInterface

Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::VectorInterface.Zero) where {T} = promote_to(TT, zero(T))
Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::VectorInterface.One) where {T} = promote_to(TT, one(T))

Reactant.promote_to(::Type{TracedRNumber}, v::VectorInterface.Zero) = promote_to(TracedRNumber{Float64}, v)
Reactant.promote_to(::Type{TracedRNumber}, v::VectorInterface.One) = promote_to(TracedRNumber{Float64}, v)

end
