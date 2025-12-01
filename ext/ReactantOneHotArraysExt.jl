module ReactantOneHotArraysExt

using GPUArraysCore: @allowscalar
using OneHotArrays: OneHotArrays, OneHotArray
using Reactant: Reactant, AnyTracedRArray, TracedRArray, TracedRNumber
using ReactantCore: ReactantCore
using Reactant.Ops: @opcall

__compatible_eltype(::Type{T}, ::Type{U}) where {T,U} = T
function __compatible_eltype(::Type{TracedRNumber{T}}, ::Type{TracedRNumber{U}}) where {T,U}
    return TracedRNumber{T}
end
__compatible_eltype(::Type{TracedRNumber{T}}, ::Type{U}) where {T,U} = T
__compatible_eltype(::Type{T}, ::Type{TracedRNumber{U}}) where {T,U} = TracedRNumber{T}

function Reactant.traced_type_inner(
    @nospecialize(_::Type{OneHotArray{T,N,Np1,I}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N,Np1,I}
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    return OneHotArray{__compatible_eltype(T, eltype(I2)),N,Np1,I2}
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

function OneHotArrays.onehotbatch(data::AnyTracedRArray{<:Any,N}, labels) where {N}
    # TODO: add checkbounds once we support that with TracedRNumber
    labels_expanded = @opcall broadcast_in_dim(
        Reactant.promote_to(
            TracedRArray{Reactant.unwrapped_eltype(labels),1},
            ReactantCore.materialize_traced_array(vec(labels)),
        ),
        Int64[1],
        [length(labels), size(data)...],
    )
    data = ReactantCore.materialize_traced_array(reshape(data, 1, size(data)...))
    indices = UInt32.(@opcall(findfirst(data .== labels_expanded; dimension=1)))
    return OneHotArray{TracedRNumber{UInt32},N,N + 1,typeof(indices)}(
        indices, length(labels)
    )
end

function OneHotArrays.onehotbatch(
    data::AnyTracedRArray{<:Integer,N}, labels::AbstractUnitRange{<:Integer}
) where {N}
    # TODO: add checkbounds once we support that with TracedRNumber
    indices = map(
        TracedRNumber{UInt32} ∘ Base.Fix2(+, 1 - first(labels)),
        ReactantCore.materialize_traced_array(data),
    )
    return OneHotArray{TracedRNumber{UInt32},N,N + 1,typeof(indices)}(
        indices, length(labels)
    )
end

function OneHotArrays.onecold(y::AnyTracedRArray{T,1}, labels=1:length(y)) where {T}
    nl = length(labels)
    ny = length(y)
    nl == ny || throw(
        DimensionMismatch(
            "onecold got $nl labels for a vector of length $ny, these must agree"
        ),
    )
    imax = argmax(y)
    # TODO: error if ymax is nan
    labels_arr = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(labels),1}, labels
    )
    return @allowscalar labels_arr[imax]
end

function OneHotArrays.onecold(y::AnyTracedRArray{T}, labels=1:size(y, 1)) where {T}
    nl = length(labels)
    ny = size(y, 1)
    nl == ny || throw(
        DimensionMismatch(
            "onecold got $nl labels for an array with first dimension of size $ny, these must agree",
        ),
    )
    labels_arr = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(labels),1}, labels
    )
    labels_expanded = @opcall broadcast_in_dim(
        labels_arr, Int64[1], Int64[nl, size(y)[2:end]...]
    )
    return ReactantCore.materialize_traced_array(
        vec(getindex(labels_expanded, argmax(y; dims=1)))
    )
end

end
