module ReactantYaoBlocksExt

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:XGate}
) where {D,T,S}
    M = Reactant.broadcast_to_size(zero(T), (2, 2))
    M[1, 1] = cos(R.theta / 2)
    M[2, 2] = cos(R.theta / 2)
    M[1, 2] = -im * sin(R.theta / 2)
    M[2, 1] = -im * sin(R.theta / 2)
    return M
end

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:YGate}
) where {D,T,S}
    M = Reactant.broadcast_to_size(zero(T), (2, 2))
    M[1, 1] = cos(R.theta / 2)
    M[2, 2] = cos(R.theta / 2)
    M[1, 2] = -sin(R.theta / 2)
    M[2, 1] = sin(R.theta / 2)
    return M
end

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:ZGate}
) where {D,T,S}
    M = Reactant.broadcast_to_size(zero(T), (2, 2))
    M[1, 1] = exp(-im * R.theta / 2)
    M[2, 2] = exp(im * R.theta / 2)
    return M
end

end
