module ReactantYaoBlocksExt

using Reactant
using Reactant.TracedUtils: broadcast_to_size
using YaoBlocks

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:XGate}
) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    c = cos(R.theta / 2)
    s = -im * sin(R.theta / 2)
    @allowscalar begin
        M[1, 1] = c
        M[2, 2] = c
        M[1, 2] = s
        M[2, 1] = s
    end
    return M
end

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:YGate}
) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    c = cos(R.theta / 2)
    s = sin(R.theta / 2)
    @allowscalar begin
        M[1, 1] = c
        M[2, 2] = c
        M[1, 2] = -s
        M[2, 1] = s
    end
    return M
end

function YaoBlocks.mat(
    ::Type{T}, R::RotationGate{D,Reactant.TracedRNumber{S},<:ZGate}
) where {D,T,S}
    M = broadcast_to_size(zero(T), (2, 2))
    x = exp(im * R.theta / 2)
    @allowscalar begin
        M[1, 1] = conj(x)
        M[2, 2] = x
    end
    return M
end

end
