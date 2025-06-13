module ReactantOneHotArraysExt

using OneHotArrays
using Reactant
using Reactant: TracedRArray, TracedRNumber, TracedUtils, Ops

function Reactant.traced_type_inner(
    @nospecialize(_::Type{OneHotArrays.OneHotArray{T,N,Np1,I}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N,Np1,I}
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    T2 = if eltype(I2) <: Reactant.TracedRNumber && !(T <: Reactant.TracedRNumber)
        Reactant.TracedRNumber{T}
    else
        T
    end
    return OneHotArrays.OneHotArray{T2,N,Np1,I2}
end

# OneHotArray is a <: AbstractArray{Bool, M} so our usual dispatches don't work
function TracedUtils.broadcast_to_size(
    r::OneHotArrays.OneHotArray{T,N,Np1,<:Reactant.TracedRArray}, rsize
) where {T,N,Np1}
    return TracedUtils.broadcast_to_size(TracedUtils.materialize_traced_array(r), rsize)
end

function TracedUtils.materialize_traced_array(r::OneHotArrays.OneHotArray)
    indices = vec(r.indices)
    N = r.nlabels
    B = length(indices)

    linear_indices =
        TracedUtils.promote_to(TracedRArray{Int64,ndims(r.indices)}, indices) .+
        Ops.iota(Int64, [B]; iota_dimension=1) .* N

    z = Ops.fill(false, (N, B))
    z[linear_indices] = fill(true, length(linear_indices))
    return reshape(z, size(r))
end

function Base.Array(
    r::OneHotArrays.OneHotArray{T,N,Np1,<:Reactant.AbstractConcreteArray}
) where {T,N,Np1}
    return Array(reshape(Array(r.indices), 1, size(r.indices)...) .== 1:(r.nlabels))
end

end
