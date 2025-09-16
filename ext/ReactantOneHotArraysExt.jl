module ReactantOneHotArraysExt

using OneHotArrays: OneHotArray
using Reactant: Reactant, TracedRArray, TracedRNumber, Ops
using ReactantCore: ReactantCore
using Reactant.Ops: @opcall

function Reactant.traced_type_inner(
    @nospecialize(_::Type{OneHotArray{T,N,Np1,I}}),
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
    return OneHotArray{T2,N,Np1,I2}
end

function ReactantCore.materialize_traced_array(r::OneHotArray)
    indices = vec(r.indices)
    N = r.nlabels
    B = length(indices)

    linear_indices = (
        Reactant.promote_to(TracedRArray, indices) .+
        @opcall(iota(Int64, [B]; iota_dimension=1)) .* N
    )

    z = @opcall(fill(false, (N, B)))
    z[linear_indices] = fill(true, length(linear_indices))
    return reshape(z, size(r))
end

Reactant._parent(r::OneHotArray) = r.indices

function Base.Array(
    r::OneHotArray{T,N,Np1,<:Reactant.AbstractConcreteArray}
) where {T,N,Np1}
    return Array(reshape(Array(r.indices), 1, size(r.indices)...) .== 1:(r.nlabels))
end

end
