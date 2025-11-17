module ReactantYaoBlocksExt

using Reactant: broadcast_to_size

import YaoBlocks: mat

function mat(::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:XGate}) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    c = cos(R.theta / 2)
    s = -im * sin(R.theta / 2)
    M[1, 1] = c
    M[2, 2] = c
    M[1, 2] = s
    M[2, 1] = s
    return M
end

function mat(::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:YGate}) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    c = cos(R.theta / 2)
    s = sin(R.theta / 2)
    M[1, 1] = c
    M[2, 2] = c
    M[1, 2] = -s
    M[2, 1] = s
    return M
end

function mat(::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:ZGate}) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    x = exp(im * R.theta / 2)
    M[1, 1] = conj(x)
    M[2, 2] = x
    return M
end

end
