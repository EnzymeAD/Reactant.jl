module ReactantPDMatsExt

using PDMats: PDMats
using Reactant: AnyTracedRVector

# PDMats' scalar accumulation loops don't trace, use broadcasted reductions.
# The single-argument-traced methods resolve the mixed cases, the
# both-arguments-traced methods resolve their ambiguity:

function PDMats.wsumsq(w::AnyTracedRVector, a::AnyTracedRVector)
    return sum(abs2.(a) .* w)
end
function PDMats.wsumsq(w::AnyTracedRVector, a::AbstractVector)
    return sum(abs2.(a) .* w)
end
function PDMats.wsumsq(w::AbstractVector, a::AnyTracedRVector)
    return sum(abs2.(a) .* w)
end

function PDMats.invwsumsq(w::AnyTracedRVector, a::AnyTracedRVector)
    return sum(abs2.(a) ./ w)
end
function PDMats.invwsumsq(w::AnyTracedRVector, a::AbstractVector)
    return sum(abs2.(a) ./ w)
end
function PDMats.invwsumsq(w::AbstractVector, a::AnyTracedRVector)
    return sum(abs2.(a) ./ w)
end

end # module ReactantPDMatsExt
