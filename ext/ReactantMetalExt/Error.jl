# Error.jl — PJRT error handling callbacks

function _error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Ptr{Cvoid}
    return C_NULL
end

function _error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(UNIMPL_MESSAGE), Val{:message}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(UNIMPL_MESSAGE)), Val{:message_size}()
    )
    return C_NULL
end

function _error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, CAPI.PJRT_Error_Code(12), Val{:code}())  # PJRT_Error_Code_UNIMPLEMENTED
    return C_NULL
end
