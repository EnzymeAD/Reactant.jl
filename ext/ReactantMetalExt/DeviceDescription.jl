mutable struct MetalDeviceDescription <: DeviceDescription
    const id::Cint
    const pid::Cint
    # name of the GPU (e.g. "Apple M1 Pro")
    const kind::String
end

Plugin.id(desc::MetalDeviceDescription) = desc.id
process_index(desc::MetalDeviceDescription) = desc.pid
kind(desc::MetalDeviceDescription) = desc.kind
Base.show(io::IO, desc::MetalDeviceDescription) = print(io, "metal: id=", desc.id)
function Base.print(io::IO, desc::MetalDeviceDescription)
    return print(io, "MetalDevice(id=", desc.id, ")")
end
attributes(::MetalDeviceDescription) = ()
