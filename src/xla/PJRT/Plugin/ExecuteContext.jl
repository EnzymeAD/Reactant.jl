"""
    ExecuteContext()

Data for a single execution, which C passes to `PJRT_LoadedExecutable_Execute` and the plugin gets in [`execute`](@ref).

`ffi_user_data` holds the FFI user data that C adds with `PJRT_FFI_UserData_Add` (see [`FFIExtension`](@ref)), by the
type id that `PJRT_FFI_Type_Register` gave to its type. It is meant for the FFI handlers run by the execution, e.g. to
get state from the framework that launched it.

C destroys it as soon as `PJRT_LoadedExecutable_Execute` returns, so a plugin that keeps running afterwards must keep a
reference to it.

Backs `PJRT_ExecuteContext` opaque type.
"""
mutable struct ExecuteContext
    const ffi_user_data::Dict{Int64,Ptr{Cvoid}}
end

ExecuteContext() = ExecuteContext(Dict{Int64,Ptr{Cvoid}}())

@pjrt_catch_return function pjrt_execute_context_create(
    args::Ptr{CAPI.PJRT_ExecuteContext_Create_Args}
)::Ptr{CAPI.PJRT_Error}
    context = root!(ExecuteContext())
    unsafe_store!(args.context, Ptr{CAPI.PJRT_ExecuteContext}(pointer_from_objref(context)))
    return ok()
end

@pjrt_catch_return function pjrt_execute_context_destroy(
    args::Ptr{CAPI.PJRT_ExecuteContext_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.context)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end
