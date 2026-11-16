mutable struct HostMemory <: Memory
    const id::Cint
    const kind::String
    const devices::Vector{Device}
end

id(mem::HostMemory) = mem.id
kind(mem::HostMemory) = mem.kind
kind_id(::HostMemory) = 0
function Base.show(io::IO, mem::HostMemory)
    return print(io, "Memory(id=", mem.id, ", kind=", mem.kind, ", platform=julia)")
end
Base.print(io::IO, ::HostMemory) = print(io, "Memory managed by Julia")
addressable_devices(mem::HostMemory) = mem.devices
