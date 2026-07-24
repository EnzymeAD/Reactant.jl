# Device.jl — PJRT device + device description callbacks

# Newer PJRT API struct not yet in auto-generated CAPI.jl
mutable struct PJRT_Device_Attributes end

struct PJRT_Device_GetAttributes_Args
    struct_size::Csize_t
    extension_start::Ptr{CAPI.PJRT_Extension_Base}
    device::Ptr{CAPI.PJRT_Device}
    attributes::Ptr{CAPI.PJRT_NamedValue}
    num_attributes::Csize_t
    device_attributes::Ptr{PJRT_Device_Attributes}
    attributes_deleter::Ptr{Cvoid}
end

function _device_get_description(
    args::Ptr{CAPI.PJRT_Device_GetDescription_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVDESC_HANDLE, Val{:device_description}())
    return C_NULL
end

function _device_is_addressable(args::Ptr{CAPI.PJRT_Device_IsAddressable_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, true, Val{:is_addressable}())
    return C_NULL
end

function _device_local_hardware_id(
    args::Ptr{CAPI.PJRT_Device_LocalHardwareId_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:local_hardware_id}())
    return C_NULL
end

function _device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, MEMORY_PTR_ARRAY, Val{:memories}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_memories}())
    return C_NULL
end

function _device_default_memory(args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, MEMORY_HANDLE, Val{:memory}())
    return C_NULL
end

function _devdesc_id(args::Ptr{CAPI.PJRT_DeviceDescription_Id_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:id}())
    return C_NULL
end

function _devdesc_process_index(
    args::Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:process_index}())
    return C_NULL
end

function _devdesc_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_attributes}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:attributes}())
    return C_NULL
end

function _device_get_attributes(args::Ptr{PJRT_Device_GetAttributes_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_attributes}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:attributes}())
    Reactant.unsafe_store_field!(args, cfn_noop_free, Val{:attributes_deleter}())
    return C_NULL
end

function _devdesc_kind(args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(DEVICE_KIND), Val{:device_kind}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(DEVICE_KIND)), Val{:device_kind_size}()
    )
    return C_NULL
end

function _devdesc_debug_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(DEVICE_DEBUG_STRING), Val{:debug_string}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(DEVICE_DEBUG_STRING)), Val{:debug_string_size}()
    )
    return C_NULL
end

function _devdesc_to_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_ToString_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(DEVICE_TO_STRING), Val{:to_string}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(DEVICE_TO_STRING)), Val{:to_string_size}()
    )
    return C_NULL
end
