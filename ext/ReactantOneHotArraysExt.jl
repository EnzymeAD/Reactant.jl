module ReactantOneHotArraysExt

using OneHotArrays
using Reactant

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
function Reactant.TracedUtils.broadcast_to_size(
    r::OneHotArrays.OneHotArray{T,N,Np1,<:Reactant.TracedRArray}, rsize
) where {T,N,Np1}
    return Reactant.TracedUtils.broadcast_to_size(
        Reactant.TracedUtils.materialize_traced_array(r), rsize
    )
end

function Reactant.TracedUtils.materialize_traced_array(
    r::OneHotArrays.OneHotArray{T,N,Np1,<:Reactant.TracedRArray}
) where {T,N,Np1}
    return reshape(r.indices, 1, size(r.indices)...) .== 1:(r.nlabels)
end

end
