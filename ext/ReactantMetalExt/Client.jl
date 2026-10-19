# Client.jl — PJRT client lifecycle callbacks

function _client_create(args::Ptr{CAPI.PJRT_Client_Create_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, CLIENT_HANDLE, Val{:client}())
    return C_NULL
end

function _client_destroy(args::Ptr{CAPI.PJRT_Client_Destroy_Args})::Ptr{Cvoid}
    return C_NULL
end

function _client_platform_name(args::Ptr{CAPI.PJRT_Client_PlatformName_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(PLATFORM_NAME), Val{:platform_name}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(PLATFORM_NAME)), Val{:platform_name_size}()
    )
    return C_NULL
end

function _client_process_index(args::Ptr{CAPI.PJRT_Client_ProcessIndex_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:process_index}())
    return C_NULL
end

function _client_platform_version(
    args::Ptr{CAPI.PJRT_Client_PlatformVersion_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(PLATFORM_VERSION), Val{:platform_version}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(PLATFORM_VERSION)), Val{:platform_version_size}()
    )
    return C_NULL
end

function _client_devices(args::Ptr{CAPI.PJRT_Client_Devices_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVICE_PTR_ARRAY, Val{:devices}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_devices}())
    return C_NULL
end

function _client_addressable_devices(
    args::Ptr{CAPI.PJRT_Client_AddressableDevices_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVICE_PTR_ARRAY, Val{:addressable_devices}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_addressable_devices}())
    return C_NULL
end

function _client_addressable_memories(
    args::Ptr{CAPI.PJRT_Client_AddressableMemories_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, MEMORY_PTR_ARRAY, Val{:addressable_memories}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_addressable_memories}())
    return C_NULL
end
