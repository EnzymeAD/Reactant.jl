"""
    ffi_register_type!(plugin::AbstractPlugin, name::String, type_id::Int64, info::CAPI.PJRT_FFI_Type_Info) -> Integer

Register the external type `name` in the FFI type registry of `plugin`, and return its type id. If `type_id` is `0`,
assign a new unique id; otherwise, check that it matches the id previously registered for `name`.

Backs `PJRT_FFI_Type_Register`.
"""
function ffi_register_type! end

struct FFITypeRegister{P<:AbstractPlugin} end

@pjrt_catch_return function (::FFITypeRegister{P})(
    args::Ptr{CAPI.PJRT_FFI_Type_Register_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    name = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.type_name)), unsafe_load(args.type_name_size)
    )
    info = unsafe_load(unsafe_load(args.type_info))
    # `type_id` is in-out: XLA passes the requested id (or `0`), and gets back the registered one
    id = ffi_register_type!(P(), name, unsafe_load(args.type_id), info)::Integer
    unsafe_store!(args.type_id, id)
    return ok()
end

"""
    ffi_register_handler!(
        plugin::AbstractPlugin, target::String, handler::Ptr{Cvoid}, platform::String,
        traits::CAPI.PJRT_FFI_Handler_TraitsBits,
    )

Register the typed FFI `handler` (an `XLA_FFI_Handler*`) of the custom call `target` for `platform` in `plugin`.

Backs `PJRT_FFI_Register_Handler`.
"""
function ffi_register_handler! end

struct FFIRegisterHandler{P<:AbstractPlugin} end

@pjrt_catch_return function (::FFIRegisterHandler{P})(
    args::Ptr{CAPI.PJRT_FFI_Register_Handler_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    target = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.target_name)), unsafe_load(args.target_name_size)
    )
    platform = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.platform_name)), unsafe_load(args.platform_name_size)
    )
    ffi_register_handler!(
        P(), target, unsafe_load(args.handler), platform, unsafe_load(args.traits)
    )
    return ok()
end

# stores the user data in the `ExecuteContext`, so it needs nothing from the plugin
@pjrt_catch_return function pjrt_ffi_user_data_add(
    args::Ptr{CAPI.PJRT_FFI_UserData_Add_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.context)
    ptr == C_NULL &&
        throw(invalid_argument("adding FFI user data requires an execute context"))
    context = unsafe_pointer_to_objref(ptr)::ExecuteContext
    user_data = unsafe_load(args.user_data)
    if haskey(context.ffi_user_data, user_data.type_id)
        throw(
            invalid_argument(
                "the execute context already has user data of type id $(user_data.type_id)"
            ),
        )
    end
    context.ffi_user_data[user_data.type_id] = user_data.data
    return ok()
end

"""
    FFIExtension(plugin::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base} = C_NULL)

Advertises `PJRT_FFI_Extension`, through which XLA registers FFI types and custom call handlers in the plugin, and adds
user data to the [`ExecuteContext`](@ref) of an execution.

`plugin` must implement [`ffi_register_type!`](@ref) and [`ffi_register_handler!`](@ref).
"""
mutable struct FFIExtension <: Extension
    const ftable::CAPI.PJRT_FFI_Extension

    # keepalive of the `@cfunction`s in `ftable`
    const type_register::Base.CFunction
    const register_handler::Base.CFunction

    function FFIExtension(
        ::P, next::Ptr{CAPI.PJRT_Extension_Base}=C_NULL
    ) where {P<:AbstractPlugin}
        type_register = @cfunction(
            $(FFITypeRegister{P}()),
            Ptr{CAPI.PJRT_Error},
            (Ptr{CAPI.PJRT_FFI_Type_Register_Args},)
        )
        register_handler = @cfunction(
            $(FFIRegisterHandler{P}()),
            Ptr{CAPI.PJRT_Error},
            (Ptr{CAPI.PJRT_FFI_Register_Handler_Args},)
        )

        #! format: off
        ftable = CAPI.PJRT_FFI_Extension(
            CAPI.PJRT_Extension_Base(csizeof(CAPI.PJRT_FFI_Extension), extension_type(FFIExtension), next),
            #= PJRT_FFI_Type_Register =# unsafe_convert(Ptr{Cvoid}, type_register),
            #= PJRT_FFI_UserData_Add =# @cfunction(pjrt_ffi_user_data_add, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_FFI_UserData_Add_Args},)),
            #= PJRT_FFI_Register_Handler =# unsafe_convert(Ptr{Cvoid}, register_handler),
        )
        #! format: on
        return new(ftable, type_register, register_handler)
    end
end

extension_type(::Type{FFIExtension}) = CAPI.PJRT_Extension_Type_FFI
