mutable struct HostDeviceDescription <: DeviceDescription
    const id::Cint
    const pid::Cint
end

function HostDeviceDescription()
    id = 0
    pid = 0
    return HostDeviceDescription(id, pid)
end

id(desc::HostDeviceDescription) = desc.id
process_index(desc::HostDeviceDescription) = desc.pid
kind(desc::HostDeviceDescription) = "julia-cpu"
Base.show(io::IO, desc::HostDeviceDescription) = print(io, "julia: id=", desc.id)
Base.print(io::IO, desc::HostDeviceDescription) = print(io, "JuliaDevice(id=", desc.id, ")")
attributes(::HostDeviceDescription) = ()
