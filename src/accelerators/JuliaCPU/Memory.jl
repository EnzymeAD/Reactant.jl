mutable struct HostMemory <: Memory
    const vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    const id::Cint
    const kind::String
    const devices::Vector{Device}

    function HostMemory(id, kind, devices)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory),
            id,
            kind,
            devices,
        )
    end
end

id(mem::HostMemory) = mem.id
kind(mem::HostMemory) = mem.kind
kind_id(::HostMemory) = 0
function Base.show(io::IO, mem::HostMemory)
    return print(io, "Memory(id=", mem.id, ", kind=", mem.kind, ", platform=julia)")
end
Base.print(io::IO, ::HostMemory) = print(io, "Memory managed by Julia")
addressable_devices(mem::HostMemory) = mem.devices
