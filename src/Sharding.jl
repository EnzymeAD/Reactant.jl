module Sharding

using ..Reactant: Reactant, XLA

struct Mesh{D}
    mesh_name::String
    device_ids::Array{Int,D}
    axis_names::NTuple{D,String}
    name_to_size::Dict{String,Int}
    name_to_dim::Dict{String,Int}

    function Mesh(
        mesh_name::String,
        devices::AbstractArray{<:XLA.Device,D},
        axis_names::NTuple{D,String},
    ) where {D}
        return Mesh(mesh_name, XLA.DeviceGetLocalDeviceId.(devices), axis_names)
    end

    function Mesh(
        mesh_name::String,
        device_ids::AbstractArray{<:Integer,D},
        axis_names::NTuple{D,String},
    ) where {D}
        @assert allunique(device_ids)
        name_to_size = Dict(
            name => Int64(size(device_ids, i)) for (i, name) in enumerate(axis_names)
        )
        name_to_dim = Dict(name => i for (i, name) in enumerate(axis_names))
        return new{D}(mesh_name, Int64.(device_ids), axis_names, name_to_size, name_to_dim)
    end
end

Base.length(mesh::Mesh) = length(mesh.device_ids)
Base.ndims(::Mesh{D}) where {D} = D
Base.size(mesh::Mesh) = size(mesh.device_ids)

abstract type AbstractSharding end

struct NoSharding <: AbstractSharding end

(::NoSharding)(x::AbstractArray) = x
(::NoSharding)(x::Number) = x

struct UnspecifiedSharding <: AbstractSharding end

(::UnspecifiedSharding)(x::AbstractArray) = x
(::UnspecifiedSharding)(::Number) = error("Sharding cannot be applied to Number")

struct NamedSharding{D1,P<:Tuple,D2} <: AbstractSharding
    mesh::Mesh{D1}
    partition_spec::P
    present_axes::Vector{String}
    is_closed::NTuple{D2,Bool}
    priority::NTuple{D2,Int}

    function NamedSharding(
        mesh::Mesh{D1},
        partition_spec::P;
        is_closed::NTuple{D2,Bool}=ntuple(
            i -> partition_spec[i] !== nothing, length(partition_spec)
        ),
        priority::NTuple{D2,Int}=ntuple(i -> 0, length(partition_spec)),
    ) where {D1,P<:Tuple,D2}
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

(::NamedSharding)(::Number) = error("Sharding cannot be applied to Number")

# TODO: Similar to `ConcreteRArray` we should have a client argument while constructing this
#       Though these issues go away once the types are merged

# XXX: multiple axes partitioning
function (sharding::NamedSharding)(x::AbstractArray)
    (; mesh, partition_spec) = sharding
    @assert length(partition_spec) == ndims(x)

    # XXX
    client = XLA.default_backend[]

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
        data[idx] = XLA.AsyncBuffer(
            XLA.ArrayFromHostBuffer(
                client,
                x[slices[CartesianIndex(slice_idx...)]...],
                XLA.ClientGetAddressableDevice(client, device_id),
            ),
            nothing,
        )
    end

    return Reactant.ShardedConcreteRArray{eltype(x),ndims(x),ndims(mesh),typeof(sharding)}(
        data, size(x), mesh, sharding
    )
end

end
