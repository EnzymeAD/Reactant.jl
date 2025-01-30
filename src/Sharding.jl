module Sharding

using ..Reactant: Reactant, XLA

#=
# TODO: make this into a tutorial: https://openxla.org/shardy/getting_started_jax

mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), (4, 2)), ("data", "model"))

samples_sharding = Sharding.NamedSharding(mesh, ("data", nothing))
w1_sharding = Sharding.NamedSharding(mesh, (nothing, "model"))
w2_sharding = Sharding.UnspecifiedSharding()
=#

struct Mesh{D}
    device_ids::Array{Int,D}
    axis_names::NTuple{D,String}

    function Mesh(
        devices::AbstractArray{<:XLA.Device,D}, axis_names::NTuple{D,String}
    ) where {D}
        return Mesh(XLA.DeviceGetLocalDeviceId.(devices), axis_names)
    end

    function Mesh(
        device_ids::AbstractArray{<:Integer,D}, axis_names::NTuple{D,String}
    ) where {D}
        @assert allunique(device_ids)
        return new{D}(Int64.(device_ids), axis_names)
    end
end

Base.length(mesh::Mesh) = length(mesh.device_ids)
Base.ndims(::Mesh{D}) where {D} = D

abstract type AbstractSharding end

struct UnspecifiedSharding <: AbstractSharding end

struct NamedSharding{M<:Mesh,P<:Tuple} <: AbstractSharding
    mesh::M
    partition_spec::P

    function NamedSharding(mesh::M, partition_spec::P) where {M<:Mesh,P<:Tuple}
        @assert length(partition_spec) == ndims(mesh)
        for p in partition_spec
            if p !== nothing
                @assert p isa String && p ∈ mesh.axis_names
            end
        end
        return new{M,P}(mesh, partition_spec)
    end
end

end
