module Sharding

using ..Reactant: Reactant, XLA

#=
# TODO: make this into a tutorial: https://openxla.org/shardy/getting_started_jax

using Reactant

mesh = Sharding.Mesh(
    "mesh", reshape(collect(Int64, 0:7), (4, 2)), ("data", "model")
)

samples_sharding = Sharding.NamedSharding(mesh, ("data", nothing))
w1_sharding = Sharding.NamedSharding(mesh, (nothing, "model"))
w2_sharding = Sharding.UnspecifiedSharding()

samples = rand(Float32, 3, 12) |> Reactant.to_rarray
w1 = rand(Float32, 4, 3) |> Reactant.to_rarray
w2 = rand(Float32, 2, 4) |> Reactant.to_rarray

# predict(samples, w1, w2) = sin.(w2 * (w1 * tanh.(samples)))
predict(samples, w1, w2) = w2 * (w1 * samples)

@code_hlo in_shardings=(samples_sharding, w1_sharding, w2_sharding) predict(samples, w1, w2)

@jit in_shardings=(samples_sharding, w1_sharding, w2_sharding) predict(samples, w1, w2)
=#

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
