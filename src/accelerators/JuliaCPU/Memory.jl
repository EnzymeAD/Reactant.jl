mutable struct JuliaMemory <: Memory
    const id::Cint
    const kind::String
    const devices::Vector{Device}
end

id(mem::JuliaMemory) = mem.id
kind(mem::JuliaMemory) = mem.kind
kind_id(::JuliaMemory) = 0
function Base.show(io::IO, mem::JuliaMemory)
    return print(io, "Memory(id=", mem.id, ", kind=", mem.kind, ", platform=julia)")
end
Base.print(io::IO, ::JuliaMemory) = print(io, "Memory managed by Julia")
addressable_devices(mem::JuliaMemory) = mem.devices
