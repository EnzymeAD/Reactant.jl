mutable struct JuliaDeviceDescription <: DeviceDescription
    const id::Cint
    const pid::Cint
end

function JuliaDeviceDescription()
    id = 0
    pid = 0
    return JuliaDeviceDescription(id, pid)
end

id(desc::JuliaDeviceDescription) = desc.id
process_index(desc::JuliaDeviceDescription) = desc.pid
kind(desc::JuliaDeviceDescription) = "julia-cpu"
Base.show(io::IO, desc::JuliaDeviceDescription) = print(io, "julia: id=", desc.id)
function Base.print(io::IO, desc::JuliaDeviceDescription)
    return print(io, "JuliaDevice(id=", desc.id, ")")
end
attributes(::JuliaDeviceDescription) = ()
