# Memory.jl — PJRT memory callbacks

function _memory_id(args::Ptr{CAPI.PJRT_Memory_Id_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:id}())
    return C_NULL
end

function _memory_kind(args::Ptr{CAPI.PJRT_Memory_Kind_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(MEMORY_KIND_STR), Val{:kind}())
    Reactant.unsafe_store_field!(args, Csize_t(length(MEMORY_KIND_STR)), Val{:kind_size}())
    return C_NULL
end

function _memory_kind_id(args::Ptr{CAPI.PJRT_Memory_Kind_Id_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Cint(0), Val{:kind_id}())
    return C_NULL
end

function _memory_debug_string(args::Ptr{CAPI.PJRT_Memory_DebugString_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(MEMORY_DEBUG_STR), Val{:debug_string}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(MEMORY_DEBUG_STR)), Val{:debug_string_size}()
    )
    return C_NULL
end

function _memory_to_string(args::Ptr{CAPI.PJRT_Memory_ToString_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(MEMORY_TO_STR), Val{:to_string}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(MEMORY_TO_STR)), Val{:to_string_size}()
    )
    return C_NULL
end

function _memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVICE_PTR_ARRAY, Val{:devices}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_devices}())
    return C_NULL
end
