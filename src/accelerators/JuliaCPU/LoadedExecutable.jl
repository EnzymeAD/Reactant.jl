# `Tag` is unique per compiled program, so that `compile` can define an `execute` method for each one
mutable struct JuliaLoadedExecutable{Tag} <: LoadedExecutable
    const executable::JuliaExecutable
    const devices::Vector{Device}
    const logical_ids::Vector{CAPI.PJRT_LogicalDeviceIds}
    const device_assignment::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}
    deleted::Bool
end

executable(loaded_exe::JuliaLoadedExecutable) = loaded_exe.executable

function Base.delete!(loaded_exe::JuliaLoadedExecutable)
    loaded_exe.deleted = true
    return nothing
end

isdeleted(loaded_exe::JuliaLoadedExecutable) = loaded_exe.deleted
addressable_devices(loaded_exe::JuliaLoadedExecutable) = loaded_exe.devices
addressable_device_logical_ids(loaded_exe::JuliaLoadedExecutable) = loaded_exe.logical_ids
device_assignment(loaded_exe::JuliaLoadedExecutable) = loaded_exe.device_assignment
