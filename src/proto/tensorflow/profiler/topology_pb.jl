import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"LogicalTopology.HostNetworkAddress", TopologyLocation, TopologyDimension
export var"LogicalTopology.LogicalDevice", Topology, var"LogicalTopology.LogicalHost"
export var"LogicalTopology.LogicalSlice", LogicalTopology


struct var"LogicalTopology.HostNetworkAddress"
    address::String
    interface_name::String
end
var"LogicalTopology.HostNetworkAddress"(;address = "", interface_name = "") = var"LogicalTopology.HostNetworkAddress"(address, interface_name)
PB.default_values(::Type{var"LogicalTopology.HostNetworkAddress"}) = (;address = "", interface_name = "")
PB.field_numbers(::Type{var"LogicalTopology.HostNetworkAddress"}) = (;address = 1, interface_name = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LogicalTopology.HostNetworkAddress"})
    address = ""
    interface_name = ""
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            address = PB.decode(d, String)
        elseif field_number == 2
            interface_name = PB.decode(d, String)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LogicalTopology.HostNetworkAddress"(address, interface_name)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LogicalTopology.HostNetworkAddress")
    initpos = position(e.io)
    !isempty(x.address) && PB.encode(e, 1, x.address)
    !isempty(x.interface_name) && PB.encode(e, 2, x.interface_name)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LogicalTopology.HostNetworkAddress")
    encoded_size = 0
    !isempty(x.address) && (encoded_size += PB._encoded_size(x.address, 1))
    !isempty(x.interface_name) && (encoded_size += PB._encoded_size(x.interface_name, 2))
    return encoded_size
end

struct TopologyLocation
    x::Int32
    y::Int32
    z::Int32
    host_x::Int32
    host_y::Int32
    host_z::Int32
    index_on_host::Int32
    global_id::Int32
end
TopologyLocation(;x = zero(Int32), y = zero(Int32), z = zero(Int32), host_x = zero(Int32), host_y = zero(Int32), host_z = zero(Int32), index_on_host = zero(Int32), global_id = zero(Int32)) = TopologyLocation(x, y, z, host_x, host_y, host_z, index_on_host, global_id)
PB.default_values(::Type{TopologyLocation}) = (;x = zero(Int32), y = zero(Int32), z = zero(Int32), host_x = zero(Int32), host_y = zero(Int32), host_z = zero(Int32), index_on_host = zero(Int32), global_id = zero(Int32))
PB.field_numbers(::Type{TopologyLocation}) = (;x = 1, y = 2, z = 3, host_x = 4, host_y = 5, host_z = 6, index_on_host = 7, global_id = 8)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TopologyLocation})
    x = zero(Int32)
    y = zero(Int32)
    z = zero(Int32)
    host_x = zero(Int32)
    host_y = zero(Int32)
    host_z = zero(Int32)
    index_on_host = zero(Int32)
    global_id = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            x = PB.decode(d, Int32)
        elseif field_number == 2
            y = PB.decode(d, Int32)
        elseif field_number == 3
            z = PB.decode(d, Int32)
        elseif field_number == 4
            host_x = PB.decode(d, Int32)
        elseif field_number == 5
            host_y = PB.decode(d, Int32)
        elseif field_number == 6
            host_z = PB.decode(d, Int32)
        elseif field_number == 7
            index_on_host = PB.decode(d, Int32)
        elseif field_number == 8
            global_id = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return TopologyLocation(x, y, z, host_x, host_y, host_z, index_on_host, global_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TopologyLocation)
    initpos = position(e.io)
    x.x != zero(Int32) && PB.encode(e, 1, x.x)
    x.y != zero(Int32) && PB.encode(e, 2, x.y)
    x.z != zero(Int32) && PB.encode(e, 3, x.z)
    x.host_x != zero(Int32) && PB.encode(e, 4, x.host_x)
    x.host_y != zero(Int32) && PB.encode(e, 5, x.host_y)
    x.host_z != zero(Int32) && PB.encode(e, 6, x.host_z)
    x.index_on_host != zero(Int32) && PB.encode(e, 7, x.index_on_host)
    x.global_id != zero(Int32) && PB.encode(e, 8, x.global_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TopologyLocation)
    encoded_size = 0
    x.x != zero(Int32) && (encoded_size += PB._encoded_size(x.x, 1))
    x.y != zero(Int32) && (encoded_size += PB._encoded_size(x.y, 2))
    x.z != zero(Int32) && (encoded_size += PB._encoded_size(x.z, 3))
    x.host_x != zero(Int32) && (encoded_size += PB._encoded_size(x.host_x, 4))
    x.host_y != zero(Int32) && (encoded_size += PB._encoded_size(x.host_y, 5))
    x.host_z != zero(Int32) && (encoded_size += PB._encoded_size(x.host_z, 6))
    x.index_on_host != zero(Int32) && (encoded_size += PB._encoded_size(x.index_on_host, 7))
    x.global_id != zero(Int32) && (encoded_size += PB._encoded_size(x.global_id, 8))
    return encoded_size
end

struct TopologyDimension
    x::Int32
    y::Int32
    z::Int32
end
TopologyDimension(;x = zero(Int32), y = zero(Int32), z = zero(Int32)) = TopologyDimension(x, y, z)
PB.default_values(::Type{TopologyDimension}) = (;x = zero(Int32), y = zero(Int32), z = zero(Int32))
PB.field_numbers(::Type{TopologyDimension}) = (;x = 1, y = 2, z = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TopologyDimension})
    x = zero(Int32)
    y = zero(Int32)
    z = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            x = PB.decode(d, Int32)
        elseif field_number == 2
            y = PB.decode(d, Int32)
        elseif field_number == 3
            z = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return TopologyDimension(x, y, z)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TopologyDimension)
    initpos = position(e.io)
    x.x != zero(Int32) && PB.encode(e, 1, x.x)
    x.y != zero(Int32) && PB.encode(e, 2, x.y)
    x.z != zero(Int32) && PB.encode(e, 3, x.z)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TopologyDimension)
    encoded_size = 0
    x.x != zero(Int32) && (encoded_size += PB._encoded_size(x.x, 1))
    x.y != zero(Int32) && (encoded_size += PB._encoded_size(x.y, 2))
    x.z != zero(Int32) && (encoded_size += PB._encoded_size(x.z, 3))
    return encoded_size
end

struct var"LogicalTopology.LogicalDevice"
    global_id::Int32
    slice_local_id::Int32
    host_local_id::Int32
end
var"LogicalTopology.LogicalDevice"(;global_id = zero(Int32), slice_local_id = zero(Int32), host_local_id = zero(Int32)) = var"LogicalTopology.LogicalDevice"(global_id, slice_local_id, host_local_id)
PB.default_values(::Type{var"LogicalTopology.LogicalDevice"}) = (;global_id = zero(Int32), slice_local_id = zero(Int32), host_local_id = zero(Int32))
PB.field_numbers(::Type{var"LogicalTopology.LogicalDevice"}) = (;global_id = 1, slice_local_id = 2, host_local_id = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LogicalTopology.LogicalDevice"})
    global_id = zero(Int32)
    slice_local_id = zero(Int32)
    host_local_id = zero(Int32)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            global_id = PB.decode(d, Int32)
        elseif field_number == 2
            slice_local_id = PB.decode(d, Int32)
        elseif field_number == 3
            host_local_id = PB.decode(d, Int32)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LogicalTopology.LogicalDevice"(global_id, slice_local_id, host_local_id)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LogicalTopology.LogicalDevice")
    initpos = position(e.io)
    x.global_id != zero(Int32) && PB.encode(e, 1, x.global_id)
    x.slice_local_id != zero(Int32) && PB.encode(e, 2, x.slice_local_id)
    x.host_local_id != zero(Int32) && PB.encode(e, 3, x.host_local_id)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LogicalTopology.LogicalDevice")
    encoded_size = 0
    x.global_id != zero(Int32) && (encoded_size += PB._encoded_size(x.global_id, 1))
    x.slice_local_id != zero(Int32) && (encoded_size += PB._encoded_size(x.slice_local_id, 2))
    x.host_local_id != zero(Int32) && (encoded_size += PB._encoded_size(x.host_local_id, 3))
    return encoded_size
end

struct Topology
    chips_per_host_bounds::Union{Nothing,TopologyDimension}
    host_bounds::Union{Nothing,TopologyDimension}
    mesh_location::Vector{TopologyLocation}
end
Topology(;chips_per_host_bounds = nothing, host_bounds = nothing, mesh_location = Vector{TopologyLocation}()) = Topology(chips_per_host_bounds, host_bounds, mesh_location)
PB.default_values(::Type{Topology}) = (;chips_per_host_bounds = nothing, host_bounds = nothing, mesh_location = Vector{TopologyLocation}())
PB.field_numbers(::Type{Topology}) = (;chips_per_host_bounds = 1, host_bounds = 2, mesh_location = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Topology})
    chips_per_host_bounds = Ref{Union{Nothing,TopologyDimension}}(nothing)
    host_bounds = Ref{Union{Nothing,TopologyDimension}}(nothing)
    mesh_location = PB.BufferedVector{TopologyLocation}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, chips_per_host_bounds)
        elseif field_number == 2
            PB.decode!(d, host_bounds)
        elseif field_number == 3
            PB.decode!(d, mesh_location)
        else
            Base.skip(d, wire_type)
        end
    end
    return Topology(chips_per_host_bounds[], host_bounds[], mesh_location[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Topology)
    initpos = position(e.io)
    !isnothing(x.chips_per_host_bounds) && PB.encode(e, 1, x.chips_per_host_bounds)
    !isnothing(x.host_bounds) && PB.encode(e, 2, x.host_bounds)
    !isempty(x.mesh_location) && PB.encode(e, 3, x.mesh_location)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Topology)
    encoded_size = 0
    !isnothing(x.chips_per_host_bounds) && (encoded_size += PB._encoded_size(x.chips_per_host_bounds, 1))
    !isnothing(x.host_bounds) && (encoded_size += PB._encoded_size(x.host_bounds, 2))
    !isempty(x.mesh_location) && (encoded_size += PB._encoded_size(x.mesh_location, 3))
    return encoded_size
end

struct var"LogicalTopology.LogicalHost"
    slice_local_id::Int32
    network_addresses::Vector{var"LogicalTopology.HostNetworkAddress"}
    devices::Vector{var"LogicalTopology.LogicalDevice"}
end
var"LogicalTopology.LogicalHost"(;slice_local_id = zero(Int32), network_addresses = Vector{var"LogicalTopology.HostNetworkAddress"}(), devices = Vector{var"LogicalTopology.LogicalDevice"}()) = var"LogicalTopology.LogicalHost"(slice_local_id, network_addresses, devices)
PB.default_values(::Type{var"LogicalTopology.LogicalHost"}) = (;slice_local_id = zero(Int32), network_addresses = Vector{var"LogicalTopology.HostNetworkAddress"}(), devices = Vector{var"LogicalTopology.LogicalDevice"}())
PB.field_numbers(::Type{var"LogicalTopology.LogicalHost"}) = (;slice_local_id = 1, network_addresses = 2, devices = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LogicalTopology.LogicalHost"})
    slice_local_id = zero(Int32)
    network_addresses = PB.BufferedVector{var"LogicalTopology.HostNetworkAddress"}()
    devices = PB.BufferedVector{var"LogicalTopology.LogicalDevice"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            slice_local_id = PB.decode(d, Int32)
        elseif field_number == 2
            PB.decode!(d, network_addresses)
        elseif field_number == 3
            PB.decode!(d, devices)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LogicalTopology.LogicalHost"(slice_local_id, network_addresses[], devices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LogicalTopology.LogicalHost")
    initpos = position(e.io)
    x.slice_local_id != zero(Int32) && PB.encode(e, 1, x.slice_local_id)
    !isempty(x.network_addresses) && PB.encode(e, 2, x.network_addresses)
    !isempty(x.devices) && PB.encode(e, 3, x.devices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LogicalTopology.LogicalHost")
    encoded_size = 0
    x.slice_local_id != zero(Int32) && (encoded_size += PB._encoded_size(x.slice_local_id, 1))
    !isempty(x.network_addresses) && (encoded_size += PB._encoded_size(x.network_addresses, 2))
    !isempty(x.devices) && (encoded_size += PB._encoded_size(x.devices, 3))
    return encoded_size
end

struct var"LogicalTopology.LogicalSlice"
    global_id::Int32
    hosts::Vector{var"LogicalTopology.LogicalHost"}
end
var"LogicalTopology.LogicalSlice"(;global_id = zero(Int32), hosts = Vector{var"LogicalTopology.LogicalHost"}()) = var"LogicalTopology.LogicalSlice"(global_id, hosts)
PB.default_values(::Type{var"LogicalTopology.LogicalSlice"}) = (;global_id = zero(Int32), hosts = Vector{var"LogicalTopology.LogicalHost"}())
PB.field_numbers(::Type{var"LogicalTopology.LogicalSlice"}) = (;global_id = 1, hosts = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"LogicalTopology.LogicalSlice"})
    global_id = zero(Int32)
    hosts = PB.BufferedVector{var"LogicalTopology.LogicalHost"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            global_id = PB.decode(d, Int32)
        elseif field_number == 2
            PB.decode!(d, hosts)
        else
            Base.skip(d, wire_type)
        end
    end
    return var"LogicalTopology.LogicalSlice"(global_id, hosts[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"LogicalTopology.LogicalSlice")
    initpos = position(e.io)
    x.global_id != zero(Int32) && PB.encode(e, 1, x.global_id)
    !isempty(x.hosts) && PB.encode(e, 2, x.hosts)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"LogicalTopology.LogicalSlice")
    encoded_size = 0
    x.global_id != zero(Int32) && (encoded_size += PB._encoded_size(x.global_id, 1))
    !isempty(x.hosts) && (encoded_size += PB._encoded_size(x.hosts, 2))
    return encoded_size
end

struct LogicalTopology
    slices::Vector{var"LogicalTopology.LogicalSlice"}
end
LogicalTopology(;slices = Vector{var"LogicalTopology.LogicalSlice"}()) = LogicalTopology(slices)
PB.default_values(::Type{LogicalTopology}) = (;slices = Vector{var"LogicalTopology.LogicalSlice"}())
PB.field_numbers(::Type{LogicalTopology}) = (;slices = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:LogicalTopology})
    slices = PB.BufferedVector{var"LogicalTopology.LogicalSlice"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, slices)
        else
            Base.skip(d, wire_type)
        end
    end
    return LogicalTopology(slices[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::LogicalTopology)
    initpos = position(e.io)
    !isempty(x.slices) && PB.encode(e, 1, x.slices)
    return position(e.io) - initpos
end
function PB._encoded_size(x::LogicalTopology)
    encoded_size = 0
    !isempty(x.slices) && (encoded_size += PB._encoded_size(x.slices, 1))
    return encoded_size
end
