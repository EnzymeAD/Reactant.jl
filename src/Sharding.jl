module Sharding

using ..Reactant: Reactant, XLA

struct Mesh{D}
    mesh_name::String
    device_ids::Array{Int,D}
    axis_names::NTuple{D,String}
    name_to_size::Dict{String,Int}

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
        return new{D}(mesh_name, Int64.(device_ids), axis_names, name_to_size)
    end
end

Base.length(mesh::Mesh) = length(mesh.device_ids)
Base.ndims(::Mesh{D}) where {D} = D

abstract type AbstractSharding end

struct NoSharding <: AbstractSharding end

struct UnspecifiedSharding <: AbstractSharding end

struct NamedSharding{D1,P<:Tuple,D2} <: AbstractSharding
    mesh::Mesh{D1}
    partition_spec::P
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
        for p in partition_spec
            if p !== nothing
                if p isa String
                    @assert p ∈ mesh.axis_names
                elseif p isa Tuple
                    for pᵢ in p
                        @assert pᵢ isa String && pᵢ ∈ mesh.axis_names
                    end
                else
                    error("Invalid partition spec $p")
                end
            end
        end
        return new{D1,P,D2}(mesh, partition_spec, is_closed, priority)
    end
end

end
