module ReactantVectorInterfaceExt

using Reactant: Reactant, TracedRNumber, promote_to
using VectorInterface: Zero, One

Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::Zero) where {T} = promote_to(TT, zero(T))
Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::One) where {T} = promote_to(TT, one(T))

Reactant.promote_to(::Type{TracedRNumber}, v::Zero) = promote_to(TracedRNumber{Float64}, v)
Reactant.promote_to(::Type{TracedRNumber}, v::One) = promote_to(TracedRNumber{Float64}, v)

end
