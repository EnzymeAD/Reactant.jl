module ReactantVectorInterfaceExt

using Rectant
using VectorInterface

function Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::VectorInterface.Zero) where {T}
    return promote_to(TT, zero(T))
end
function Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::VectorInterface.One) where {T}
    return promote_to(TT, one(T))
end

function Reactant.promote_to(::Type{TracedRNumber}, v::VectorInterface.Zero)
    return promote_to(TracedRNumber{Float64}, v)
end
function Reactant.promote_to(::Type{TracedRNumber}, v::VectorInterface.One)
    return promote_to(TracedRNumber{Float64}, v)
end

end
