module Sharding

using ..Reactant: Reactant, XLA

# NOTE: PjRt doesn't provide a native sharding mechanism, so this file implements sharding
#       at the julia level. With our migration to IFRt, we should be able to rewrite this
#       logic to directly use the sharded arrays from IFRt. This would also simplify our
#       logic of storing multiple arrays in ConcreteRArray struct

struct Mesh{D}
    device_ids::Array{Int,D}
    axis_names::NTuple{D,String}
    name_to_size::Dict{String,Int}
    name_to_dim::Dict{String,Int}

    function Mesh(devices::AbstractArray{<:XLA.Device}, axis_names)
        return Mesh(XLA.DeviceGetLocalDeviceId.(devices), axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D}, axis_names::NTuple{D,String}
    ) where {D}
        @assert allunique(device_ids)
        name_to_size = Dict(
            name => Int64(size(device_ids, i)) for (i, name) in enumerate(axis_names)
        )
        name_to_dim = Dict(name => i for (i, name) in enumerate(axis_names))
        return new{D}(Int64.(device_ids), axis_names, name_to_size, name_to_dim)
    end
end

Base.length(mesh::Mesh) = length(mesh.device_ids)
Base.ndims(::Mesh{D}) where {D} = D
Base.size(mesh::Mesh) = size(mesh.device_ids)

abstract type AbstractSharding end

function (T::AbstractSharding)(::XLA.Client, device, ::AbstractArray)
    return error("(::$(T))(::XLA.Client, ::AbstractArray) is not implemented")
end

struct NoSharding <: AbstractSharding end

finalized_sharding(::Type{NoSharding}) = FinalizedNoSharding

# This allows us to mark entire branches as NoSharding
Base.getproperty(::NoSharding, x) = NoSharding()

function (::NoSharding)(client::XLA.Client, device, x::AbstractArray)
    buffer = XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, x, device), nothing)
    return [buffer], FinalizedNoSharding()
end

struct NamedSharding{D1,P<:Tuple,D2} <: AbstractSharding
    mesh::Mesh{D1}
    partition_spec::P
    present_axes::Vector{String}
    is_closed::NTuple{D2,Bool}
    priority::NTuple{D2,Int}

    function NamedSharding(
        mesh::Mesh{D1},
        partition_spec::P;
        is_closed::NTuple{D2,Bool}=ntuple(Returns(true), length(partition_spec)),
        # negative priority means that priority is not considered by shardy
        priority::NTuple{D2,Int}=ntuple(i -> -1, length(partition_spec)),
    ) where {D1,P<:Tuple,D2}
        # TODO: we need to check how open sharding works in XLA, i.e. how to specify inputs
        @assert all(is_closed) "All partitions must be closed for now."
        present_axes = String[]
        for p in partition_spec
            if p !== nothing
                if p isa String
                    @assert p ∈ mesh.axis_names
                    push!(present_axes, p)
                elseif p isa Tuple
                    for pᵢ in p
                        @assert pᵢ isa String && pᵢ ∈ mesh.axis_names
                        push!(present_axes, pᵢ)
                    end
                else
                    error("Invalid partition spec $p")
                end
            end
        end
        @assert length(unique(present_axes)) == length(present_axes) "Duplicate axis names!"
        return new{D1,P,D2}(mesh, partition_spec, present_axes, is_closed, priority)
    end
end

finalized_sharding(::Type{<:NamedSharding}) = FinalizedNamedSharding

# XXX: multiple axes partitioning
function (sharding::NamedSharding)(client::XLA.Client, ::Nothing, x::AbstractArray)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    # Fast Path for replicating the input across all devices
    if all(Base.Fix2(===, nothing), partition_spec)
        data = Array{XLA.AsyncBuffer,ndims(mesh)}(undef, size(mesh))
        device_to_array_slices = Array{Vector{UnitRange{Int64}},ndims(mesh)}(
            undef, size(mesh)
        )

        for idx in CartesianIndices(data)
            device_id = mesh.device_ids[idx]
            device_to_array_slices[idx] = [1:size(x, i) for i in 1:ndims(x)]
            data[idx] = XLA.AsyncBuffer(
                XLA.ArrayFromHostBuffer(client, x, device_id), nothing
            )
        end

        return (
            data,
            FinalizedNamedSharding{typeof(sharding),ndims(mesh)}(
                sharding, device_to_array_slices
            ),
        )
    end

    ndevices = Vector{Int}(undef, ndims(x))
    axis_name_to_dim_and_offset = Dict{String,Tuple{Int,Int}}()
    for i in 1:ndims(x)
        p = partition_spec[i]
        if p === nothing
            ndevices[i] = 1
        else
            if p isa Tuple
                offset = 0
                for pᵢ in p
                    axis_name_to_dim_and_offset[pᵢ] = (i, offset)
                    offset += mesh.name_to_size[pᵢ]
                end
                ndevices[i] = offset
            else
                axis_name_to_dim_and_offset[p] = (i, 0)
                ndevices[i] = mesh.name_to_size[p]
            end
        end
    end

    for (sz, ndevice) in zip(size(x), ndevices)
        @assert sz % ndevice == 0 "$(size(x)) must be divisible by $(ndevices)"
    end
    strides = size(x) .÷ ndevices

    slices = Array{Vector{UnitRange{Int64}},ndims(x)}(undef, Tuple(ndevices))

    for idx in CartesianIndices(slices)
        idx_tup = Tuple(idx)
        slices[idx] = [
            (i1 + 1):i2 for (i1, i2) in zip((idx_tup .- 1) .* strides, idx_tup .* strides)
        ]
    end

    data = Array{XLA.AsyncBuffer,ndims(mesh)}(undef, size(mesh))
    device_to_array_slices = Array{Vector{UnitRange{Int64}},ndims(mesh)}(undef, size(mesh))

    for idx in CartesianIndices(data)
        device_id = mesh.device_ids[idx]
        idx_tup = Tuple(idx)
        slice_idx = ones(Int, ndims(slices))
        for (axis_name, idxᵢ) in zip(mesh.axis_names, idx_tup)
            if haskey(axis_name_to_dim_and_offset, axis_name)
                dim, offset = axis_name_to_dim_and_offset[axis_name]
                slice_idx[dim] = idxᵢ + offset
            end
        end
        device_to_array_slices[idx] = slices[CartesianIndex(slice_idx...)]
        data[idx] = XLA.AsyncBuffer(
            XLA.ArrayFromHostBuffer(
                client,
                x[device_to_array_slices[idx]...],
                XLA.ClientGetAddressableDevice(
                    client, XLA.device_ordinal(client, device_id)
                ),
            ),
            nothing,
        )
    end

    return (
        data,
        FinalizedNamedSharding{typeof(sharding),ndims(mesh)}(
            sharding, device_to_array_slices
        ),
    )
end

# Internal Type that mimics XYZSharding but contains mapping from device to array slices
abstract type AbstractFinalizedSharding <: AbstractSharding end

function Base.getproperty(sharding::AbstractFinalizedSharding, name::Symbol)
    name ∈ (:sharding, :device_to_array_slices) && return getfield(sharding, name)
    return getfield(sharding.sharding, name)
end

function (sharding::AbstractFinalizedSharding)(client::XLA.Client, device, x::AbstractArray)
    return (sharding.sharding)(client, device, x)
end

struct FinalizedNoSharding <: AbstractFinalizedSharding end

function Base.getproperty(::FinalizedNoSharding, name::Symbol)
    @assert name === :sharding
    return NoSharding()
end

struct FinalizedNamedSharding{S<:NamedSharding,D} <: AbstractFinalizedSharding
    sharding::S
    device_to_array_slices::Array{Vector{UnitRange{Int64}},D}
end

"""
    is_sharded(sharding)
    is_sharded(x::AbstractArray)

Checks whether the given sharding refers to no sharding.
"""
is_sharded(::NoSharding) = false
is_sharded(::FinalizedNoSharding) = false
is_sharded(::NamedSharding) = true
is_sharded(::FinalizedNamedSharding) = true

function Sharding.is_sharded(x::AbstractArray)
    ancestor_x = Reactant.ancestor(x)
    if hasfield(typeof(ancestor_x), :sharding)
        return is_sharded(ancestor_x.sharding)
    end
    return false
end

end
