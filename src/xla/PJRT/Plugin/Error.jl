"""
    Error(code::CAPI.PJRT_Error_Code, message)

A `PJRT_Error` carrying an error `code` and a human-readable `message`.

Throw it from a `@pjrt_catch_return` callback to return it to C.

!!! info
Success is a null `PJRT_Error*` (`ok()`), not an `Error` with code `OK`.

!!! warning
It is GC rooted when handed to C and unrooted by `PJRT_Error_Destroy`.
Since every receiver destroys its own error, the same `Error` must never be handed out twice.
"""
mutable struct Error
    const vtable::Ptr{CAPI.PJRT_Error_FunctionTable}
    const code::CAPI.PJRT_Error_Code
    const message::String

    function Error(code::CAPI.PJRT_Error_Code, message)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Error_FunctionTable}, vtable_error), code, message
        )
    end
end

# build an `Error` from the status fields of a C struct; the message is owned by C, so it's copied
function Error(code::CAPI.PJRT_Error_Code, message::Cstring, message_size::Integer)
    ptr = Ptr{UInt8}(message)
    return Error(code, ptr == C_NULL ? "" : unsafe_string(ptr, message_size))
end

# XLA considers `nullptr` to be OK as it avoids allocations
ok() = C_NULL

"""
    unimplemented(reason="this feature is not implemented yet!") -> Error

Return an `UNIMPLEMENTED` [`Error`](@ref) with message `reason`, for features that are not supported yet.

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function unimplemented(reason="this feature is not implemented yet!")
    return Error(CAPI.PJRT_Error_Code_UNIMPLEMENTED, reason)
end

"""
    invalid_argument(reason="invalid argument") -> Error

Return an `INVALID_ARGUMENT` [`Error`](@ref) with message `reason`, for arguments that are malformed or that refer to
an object in the wrong state (e.g. a deleted buffer).

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function invalid_argument(reason="invalid argument")
    return Error(CAPI.PJRT_Error_Code_INVALID_ARGUMENT, reason)
end

"""
    resource_exhausted(reason="out of memory") -> Error

Return a `RESOURCE_EXHAUSTED` [`Error`](@ref) with message `reason`, for failed allocations.

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function resource_exhausted(reason="out of memory")
    return Error(CAPI.PJRT_Error_Code_RESOURCE_EXHAUSTED, reason)
end

# these are required by CAPI.PJRT_Api
function pjrt_error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    pjrt_error_destroy_obj(error)
    return nothing
end

function pjrt_error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    pjrt_error_message_obj(error, args.message, args.message_size)
    return nothing
end

function pjrt_error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    code = pjrt_error_getcode_obj(error)
    unsafe_store!(args.code, code)
    return ok()
end

function pjrt_error_for_each_payload(
    args::Ptr{CAPI.PJRT_Error_ForEachPayload_Args}
)::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    visitor = unsafe_load(args.visitor)
    user_arg = unsafe_load(args.user_arg)
    pjrt_error_for_each_payload_obj(error, visitor, user_arg)
    return ok()
end

# this are required by CAPI.PJRT_Error_FunctionTable
function pjrt_error_destroy_obj(error::Ptr{Error})::Cvoid
    error != C_NULL && unroot!(error)
    return nothing
end

function pjrt_error_message_obj(
    error::Ptr{Error}, message::Ptr{Cstring}, message_size::Ptr{Csize_t}
)::Cvoid
    error == C_NULL && return nothing
    err = unsafe_pointer_to_objref(error)::Error
    msg = err.message
    unsafe_store!(message, Cstring(pointer(msg)))
    unsafe_store!(message_size, sizeof(msg))
    return nothing
end

function pjrt_error_getcode_obj(error::Ptr{Error})::CAPI.PJRT_Error_Code
    error == C_NULL && return CAPI.PJRT_Error_Code_OK
    err = unsafe_pointer_to_objref(error)::Error
    return err.code
end

function pjrt_error_for_each_payload_obj(
    error::Ptr{Error}, visitor::CAPI.PJRT_Error_PayloadVisitor, user_arg::Ptr{Cvoid}
)::Cvoid
    # do nothing
    return nothing
end
