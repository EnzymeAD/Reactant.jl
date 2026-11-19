mutable struct MetalMemory <: Memory
    const id::Cint
    const kind::String
    const devices::Vector{Device}
end

Plugin.id(mem::MetalMemory) = mem.id
kind(mem::MetalMemory) = mem.kind
kind_id(::MetalMemory) = 0
function Base.show(io::IO, mem::MetalMemory)
    return print(io, "Memory(id=", mem.id, ", kind=", mem.kind, ", platform=metal)")
end
Base.print(io::IO, ::MetalMemory) = print(io, "Memory managed by Metal")
addressable_devices(mem::MetalMemory) = mem.devices
