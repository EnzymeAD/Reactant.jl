module ReactantVectorInterfaceExt

using Reactant: Reactant, TracedRArray, TracedRNumber, promote_to
using Reactant.TracedUtils: get_mlir_data, set_mlir_data!
using VectorInterface: VectorInterface, Zero, One

Reactant.TracedRNumber{T}(::Zero) where {T} = promote_to(TracedRNumber{T}, Zero())
Reactant.TracedRNumber{T}(::One) where {T} = promote_to(TracedRNumber{T}, One())

Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::Zero) where {T} = promote_to(TT, zero(T))
Reactant.promote_to(TT::Type{TracedRNumber{T}}, ::One) where {T} = promote_to(TT, one(T))

Reactant.promote_to(::Type{TracedRNumber}, v::Zero) = promote_to(TracedRNumber{Float64}, v)
Reactant.promote_to(::Type{TracedRNumber}, v::One) = promote_to(TracedRNumber{Float64}, v)

function Base.promote_rule(::Type{One}, ::Type{TracedRNumber{T}}) where {T<:Number}
    return TracedRNumber{T}
end
function Base.promote_rule(::Type{Zero}, ::Type{TracedRNumber{T}}) where {T<:Number}
    return TracedRNumber{T}
end

function VectorInterface.scale(x::Number, α::TracedRNumber)
    return ifelse(iszero(α), zero(x), x) * α
end

function VectorInterface.add(y::Number, x::Number, α::TracedRNumber, ::One)
    return ifelse(iszero(α), muladd(zero(x), α, y), muladd(x, α, y))
end

VectorInterface.scale(x::TracedRArray, α::Number) = VectorInterface.scale.(x, α)
VectorInterface.scale!!(x::TracedRArray, α::Number) = VectorInterface.scale!(x, α)
function VectorInterface.scale!(x::TracedRArray, α::Number)
    y = VectorInterface.scale(x, α)
    set_mlir_data!(x, get_mlir_data(y))
    return x
end

function VectorInterface.add(y::TracedRArray, x::TracedRArray, α::Number, β::Number)
    return VectorInterface.add.(y, x, α, β)
end
function VectorInterface.add!!(y::TracedRArray, x::TracedRArray, α::Number, β::Number)
    return VectorInterface.add!(y, x, α, β)
end
function VectorInterface.add!(y::TracedRArray, x::TracedRArray, α::Number, β::Number)
    z = VectorInterface.add(y, x, α, β)
    set_mlir_data!(y, get_mlir_data(z))
    return y
end

end
